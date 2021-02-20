import torch
import torch.nn as nn
import typing
from functools import reduce
from operator import mul
from . import prune
from enum import IntEnum
import sys

__all__ = ['PruningPlan', 'Dependency', 'DependencyGraph']

TORCH_CONV = nn.modules.conv._ConvNd
TORCH_BATCHNORM = nn.modules.batchnorm._BatchNorm
TORCH_PRELU = nn.PReLU
TORCH_LINEAR = nn.Linear

class OPTYPE(IntEnum):
    CONV = 0
    BN = 1
    LINEAR = 2
    PRELU = 3
    GROUP_CONV = 4
    CONCAT = 5
    SPLIT = 6
    ELEMENTWISE = 7

def _get_module_type(module):
    if isinstance(module, TORCH_CONV):
        if module.groups>1:
            return OPTYPE.GROUP_CONV
        else:
            return OPTYPE.CONV
    elif isinstance(module, TORCH_BATCHNORM):
        return OPTYPE.BN
    elif isinstance(module, TORCH_PRELU):
        return OPTYPE.PRELU
    elif isinstance(module, TORCH_LINEAR):
        return OPTYPE.LINEAR
    elif isinstance(module, _ConcatOp):
        return OPTYPE.CONCAT
    elif isinstance(module, _SplitOP):
        return OPTYPE.SPLIT
    else:
        return OPTYPE.ELEMENTWISE

def _get_node_out_channel(node):
    # get the number of output channel of a given node
    if node.type == OPTYPE.CONV or node.type == OPTYPE.GROUP_CONV:
        return node.module.out_channels
    elif node.type == OPTYPE.BN:
        return node.module.num_features
    elif node.type == OPTYPE.LINEAR:
        return node.module.out_features
    elif node.type == OPTYPE.PRELU:
        if node.module.num_parameters == 1:
            return None
        else:
            return node.module.num_parameters
    else:
        return None

def _get_node_in_channel(node):
    # get the number of input channel of a given node
    if node.type == OPTYPE.CONV or node.type == OPTYPE.GROUP_CONV:
        return node.module.in_channels
    elif node.type == OPTYPE.BN:
        return node.module.num_features
    elif node.type == OPTYPE.LINEAR:
        return node.module.in_features
    elif node.type == OPTYPE.PRELU:
        if node.module.num_parameters == 1:
            return None
        else:
            return node.module.num_parameters
    else:
        return None

# Dummy Pruning fn
def _prune_concat(layer, *args, **kargs):
    return layer, 0

def _prune_split(layer, *args, **kargs):
    return layer, 0

def _prune_elementwise_op(layer, *args, **kargs):
    return layer, 0

# Dummy module
class _ConcatOp(nn.Module):
    def __init__(self):
        super(_ConcatOp, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_ConcatOp(%s)"%(self.offsets)

class _SplitOP(nn.Module):
    def __init__(self):
        super(_SplitOP, self).__init__()
        self.offsets = None

    def __repr__(self):
        return "_SplitOP(%s)"%(self.offsets)

class _ElementWiseOp(nn.Module):
    def __init__(self):
        super(_ElementWiseOp, self).__init__()

    def __repr__(self):
        return "_ElementWiseOp()"



class _FlattenIndexTransform(object):
    def __init__(self, stride=1, reverse=False):
        self._stride = stride
        self.reverse = reverse

    def __call__(self, idxs):
        new_idxs = []
        if self.reverse is True:
            for i in idxs:
                new_idxs.append(i//self._stride)
                new_idxs = list(set(new_idxs))
        else:
            for i in idxs:
                new_idxs.extend(list(range(i*self._stride, (i+1)*self._stride)))
        return new_idxs

class _ConcatIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i-self.offset[0] for i in idxs if (i>=self.offset[0] and i<self.offset[1])]
        else:
            new_idxs = [i+self.offset[0] for i in idxs]
        return new_idxs

class _SplitIndexTransform(object):
    def __init__(self, offset, reverse=False):
        self.offset = offset
        self.reverse = reverse

    def __call__(self, idxs):
        if self.reverse == True:
            new_idxs = [i+self.offset[0] for i in idxs]
        else:
            new_idxs = [i-self.offset[0] for i in idxs if (i>=self.offset[0] and i<self.offset[1])]
        return new_idxs

class Node(object):
    def __init__(self, module, grad_fn, node_name=None):
        self.module = module
        self.grad_fn = grad_fn
        self.inputs = []  # input nodes
        self.outputs = []  # output nodes
        self.dependencies = []
        self._node_name = node_name
        self.type = _get_module_type(module)

    @property
    def node_name(self):
        return "%s (%s)" % (self._node_name, str(self.module)) if self._node_name is not None else str(self.module)

    def add_input(self, node):
        if node not in self.inputs:
            self.inputs.append(node)

    def add_output(self, node):
        if node not in self.outputs:
            self.outputs.append(node)

    def __repr__(self):
        return "<Node: (%s, %s)>" % (self.node_name, self.grad_fn)

    def __str__(self):
        return "<Node: (%s, %s)>" % (self.node_name, self.grad_fn)

    def details(self):
        fmt = "<Node: (%s, %s)>\n" % (self.node_name, self.grad_fn)
        fmt += ' '*4+'IN:\n'
        for in_node in self.inputs:
            fmt += ' '*8 + '%s\n' % (in_node)
        fmt += ' '*4+'OUT:\n'
        for out_node in self.outputs:
            fmt += ' '*8 + '%s\n' % (out_node)

        fmt += ' '*4+'DEP:\n'
        for dep in self.dependencies:
            fmt += ' '*8 + "%s\n" % (dep)
        return fmt

class Dependency(object):
    def __init__(self, trigger, handler, broken_node: Node, index_transform: typing.Callable = None):
        """
            Dependency is used to describe the dependencies between 2 modules, i.e. how the `handler`
            handles the number of channels in `broken_node.module` according to the changes `trigger`
            will make (have made) to the number of channels in the next (previous) module.

            Parameters:
                trigger (Callable or None): a pruning function which will break the dependency
                handler (Callable): a function to prune `broken_node.module`
                broken_node (Node): the layer where the dependency is changed
        """
        self.trigger = trigger
        self.handler = handler
        self.broken_node = broken_node
        self.index_transform = index_transform

    def __call__(self, idxs: list, dry_run: bool = False):
        result = self.handler(self.broken_node.module, idxs, dry_run=dry_run)  # prune the module `self.broken_node.module` using function `self.handler`
        return result

    def __repr__(self):
        return str(self)

    def __str__(self):
        return "<DEP: %s => %s on %s>" % ("None" if self.trigger is None else self.trigger.__name__, self.handler.__name__, self.broken_node.node_name)

    def is_triggered_by(self, pruning_fn):
        return pruning_fn == self.trigger

    def __eq__(self, other):
        return (self.trigger == other.trigger and self.handler == other.handler and self.broken_node == other.broken_node)

class PruningPlan(object):
    """
        Pruning plan.

        Args:
            dry_run (bool or None): default False. If True, only return the info about pruning.
            module_to_name (dict): mapping nn.module to a readable name. It will be filled by DependencyGraph.
    """

    def __init__(self):
        self._plans = list()

    def add_plan(self, dep, idxs):
        self._plans.append((dep, idxs))

    @property
    def plan(self):
        return self._plans

    def exec(self, dry_run=False):
        num_pruned = 0
        for dep, idxs in self._plans:
            print("exec:", dep)
            _, n = dep(idxs, dry_run=dry_run)  # execute the pruning
            num_pruned += n
        return num_pruned

    def has_dep(self, dep):
        for _dep, _ in self._plans:
            if dep == _dep:
                return True
        return False

    def has_pruning_op(self, dep, idxs):
        for _dep, _idxs in self._plans:
            if _dep.broken_node == dep.broken_node and _dep.handler == dep.handler and _idxs == idxs:
                return True
        return False

    def add_plan_and_merge(self, dep, idxs):
        """
            Add a new pruning plan or merge the additional indices to a existed plan
        """
        for i, (_dep, _idxs) in enumerate(self._plans):
            if _dep.broken_node == dep.broken_node and _dep.handler == dep.handler:
                self._plans[i] = (_dep, list(set(_idxs+idxs)))
                return
        self.add_plan(dep, idxs)

    def __str__(self):
        fmt = ""
        fmt += "\n-------------\n"
        totally_pruned = 0
        for dep, idxs in self._plans:
            _, n_pruned = dep(idxs, dry_run=True)
            totally_pruned += n_pruned
            fmt += "[%s, Index=%s, NumPruned=%d]\n" % (dep, idxs, n_pruned)
        fmt += "%d parameters will be pruned\n" % (totally_pruned)
        fmt += "-------------\n"
        return fmt


class DependencyGraph(object):
    """
        Properties:
            - PRUNABLE_MODULES: tuple of `nn.Module` subclasses which are prunable.
            - HANDLER: a dict with form "{optype: (in_func, out_func)}", where `in_func` refers to a function that
                prune the input channel of a certain module; `out_func` refers to a function that prune the output
                channel of a certain module; and `optype` refers to the type of operation of a certain module
            - INPUT_NODE_RULES: a dict with form "{(node.optype, in_node.optype): (in_func_node, out_func_in_node)}",
                where `node.optype` and `in_node.optype` refer to the type of operation of the current node and its
                input node respectively; `in_func_node` refers to the function that prune the input channel of the
                module in the current node and `out_func_in_node` refers to the function that prune the output channel
                of the module in the input node.
                Thus, every key of `INPUT_NODE_RULES` represents a connection form of two modules, and its value
                descibes the pruning function applied to the two modules.
                In practice, `in_func_node` acts as a `trigger` and `out_func_in_node` acts as the `handler` to
                prune `in_node` in the dependency between two nodes and (see meth:`_build_dependency` below and
                class:`Dependency` above), which means each item is used to prune the out_channel of the input node.
            - OUTPUT_NODE_RULES: similar to `INPUT_NODE_RULES`
    """
    PRUNABLE_MODULES = (nn.modules.conv._ConvNd, nn.modules.batchnorm._BatchNorm, nn.Linear, nn.PReLU)
    HANDLER = {                        # func to prune in_channel     # func to prune out_channel
                OPTYPE.CONV          :  (prune.prune_related_conv,      prune.prune_conv),
                OPTYPE.BN            :  (prune.prune_batchnorm,         prune.prune_batchnorm),
                OPTYPE.PRELU         :  (prune.prune_prelu,             prune.prune_prelu),
                OPTYPE.LINEAR        :  (prune.prune_related_linear,    prune.prune_linear),
                OPTYPE.GROUP_CONV    :  (prune.prune_group_conv,        prune.prune_group_conv),
                OPTYPE.CONCAT        :  (_prune_concat,                 _prune_concat),
                OPTYPE.SPLIT         :  (_prune_split,                  _prune_split),
                OPTYPE.ELEMENTWISE   :  (_prune_elementwise_op,         _prune_elementwise_op),
               }
    OUTPUT_NODE_RULES = {}
    INPUT_NODE_RULES = {}
    for t1 in HANDLER.keys():
        for t2 in HANDLER.keys():
            OUTPUT_NODE_RULES[(t1, t2)] = (HANDLER[t1][1], HANDLER[t2][0])
            INPUT_NODE_RULES[(t1, t2)] = (HANDLER[t1][0], HANDLER[t2][1])

    def build_dependency(self, model:torch.nn.Module, example_inputs:torch.Tensor, output_transform: callable = None, verbose: bool = True):
        self.verbose = verbose
        self._module_to_name = {module: name for (name, module) in model.named_modules()}  # to get the name of a module to generate its `Node`
        # build dependency graph
        self.module_to_node = self._obtain_forward_graph(model, example_inputs, output_transform=output_transform)
        self._build_dependency(self.module_to_node)
        # for m, node in self.module_to_node.items():
        #     print(node.details())
        self.update_index()
        return self

    def update_index(self):
        # for the ops such as flatten, concatenation, and split, indices need to be updated
        for module, node in self.module_to_node.items():
            if node.type == OPTYPE.LINEAR:
                self._set_fc_index_transform(node)
            elif node.type == OPTYPE.CONCAT:
                self._set_concat_index_transform(node)
            elif node.type == OPTYPE.SPLIT:
                self._set_split_index_transform(node)

    def get_pruning_plan(self, module, pruning_fn, idxs):
        if isinstance(module, TORCH_CONV) and module.groups > 1:
            pruning_fn = prune.prune_group_conv

        self.update_index()
        plan = PruningPlan()
        # the user pruning operation
        root_node = self.module_to_node[module]
        plan.add_plan(Dependency(pruning_fn, pruning_fn, root_node), idxs)
        visited = set()
        # print("getting plan based on {}".format(root_node.details()))

        def _fix_denpendency_graph(node, fn, indices):
            # print("==> fixing dep of {}, pruning_fn: {}".format(self._module_to_name.get(node.module, node.grad_fn), fn))
            visited.add(node)
            for dep in node.dependencies:
                # print("\t"+str(dep))
                if dep.is_triggered_by(fn):  # and dep.broken_node not in visited:
                    # print("\tis_triggered_by{}".format(fn))
                    if dep.index_transform is not None:
                        new_indices = dep.index_transform(indices)
                    else:
                        new_indices = indices
                    if len(new_indices) == 0:
                        continue
                    if dep.broken_node in visited and plan.has_pruning_op(dep, new_indices):
                        # print("\tdep.broken_node in visited and plan.has_pruning_op(dep, new_indices)\n")
                        continue
                    else:
                        plan.add_plan(dep, new_indices)
                        # print("\tadd {} to {}".format(dep, self._module_to_name.get(node.module, node.grad_fn)))
                        _fix_denpendency_graph(dep.broken_node, dep.handler, new_indices)
                else:
                    # print("\tis NOT triggered by{}, pass".format(fn))
                    pass
        _fix_denpendency_graph(root_node, pruning_fn, idxs)
        # print("="*64+"\n\n")

        # merge pruning ops
        merged_plan = PruningPlan()
        for dep, idxs in plan.plan:
            merged_plan.add_plan_and_merge(dep, idxs)
        return merged_plan

    def _build_dependency(self, module_to_node):
        for module, node in module_to_node.items():
            for in_node in node.inputs:
                # add node--in_node dependencies
                in_node_rule = self.INPUT_NODE_RULES.get((node.type, in_node.type), None)
                if in_node_rule is not None:
                    dep = Dependency(trigger=in_node_rule[0], handler=in_node_rule[1], broken_node=in_node)
                    node.dependencies.append(dep)
            for out_node in node.outputs:
                # add node--out_node dependencies
                out_node_rule = self.OUTPUT_NODE_RULES.get((node.type, out_node.type), None)
                if out_node_rule is not None:
                    dep = Dependency(trigger=out_node_rule[0], handler=out_node_rule[1], broken_node=out_node)
                    node.dependencies.append(dep)

    def _obtain_forward_graph(self, model, example_inputs, output_transform):
        """
            Build a graph by tracing back along `grad_fn` throughout the network and attaching a `Node` to each module
        """
        # module_to_node = {m: Node(m) for m in model.modules() if isinstance(m, self.PRUNABLE_MODULES)}
        model.eval().cpu()
        grad_fn_to_module = {}  # to get the corresponding module while tracing back along `grad_fn`
        visited = {}  # to generate `reused`

        def _record_module_grad_fn(module, inputs, outputs):
            if module not in visited:
                visited[module] = 1
            else:
                visited[module] += 1
            grad_fn_to_module[outputs.grad_fn] = module

        hooks = [m.register_forward_hook(_record_module_grad_fn) for m in model.modules() if isinstance(m, self.PRUNABLE_MODULES)]
        out = model(example_inputs)
        for hook in hooks:
            hook.remove()
        reused = [m for (m, count) in visited.items() if count > 1]
        module_to_node = {}
        self.inner = 0
        file = open("./_build_gragh.txt", "w")

        def _build_graph(grad_fn):
            # trace back along the given `grad_fn` and attach a node to it
            module = grad_fn_to_module.get(grad_fn, None)
            if module is not None and module in module_to_node and module not in reused:
                # ? what if module is in `reused` and `module_to_node`?
                file.write("\t"*self.inner+str(module_to_node[module])+"\n")
                for f in grad_fn.next_functions:
                    file.write("\t"*(self.inner+1)+str(module_to_node[grad_fn_to_module[f[0]]])+"\n")
                return module_to_node[module]

            if module is None:
                # which means the module is not a standard subclass of `nn.Module`, but others such as "add", customized, ect.
                # so it need to be assigned a dummy module so that #?
                if not hasattr(grad_fn, 'name'):
                    module = _ElementWiseOp()  # skip customized modules
                    if self.verbose:
                        print("[Warning] Unrecognized operation: %s. It will be treated as element-wise op" % (str(grad_fn)))
                elif 'catbackward' in grad_fn.name().lower():
                    module = _ConcatOp()
                elif 'splitbackward' in grad_fn.name().lower():
                    module = _SplitOP()
                else:
                    module = _ElementWiseOp()  # All other ops are treated as element-wise ops
                grad_fn_to_module[grad_fn] = module

            # if `module` is not in `module_to_node`, we need to add it to `module_to_node`
            # otherwise we need to get the corresponding node
            if module not in module_to_node:
                node = Node(module, grad_fn, self._module_to_name.get(module, None))
                module_to_node[module] = node
            else:
                node = module_to_node[module]
            file.write("\t"*self.inner+str(node)+"\n")
            if hasattr(grad_fn, 'next_functions'):
                self.inner += 1
                for f in grad_fn.next_functions:
                    if f[0] is not None:
                        if hasattr(f[0], 'name') and 'accumulategrad' in f[0].name().lower():  # skip leaf variables
                            file.write("\t"*self.inner+str(f[0])+"\n")
                            continue
                        input_node = _build_graph(f[0])
                        node.add_input(input_node)
                        input_node.add_output(node)
                    else:
                        file.write("\t"*self.inner+"grad_fn: None"+"\n")
                self.inner -= 1
            return node

        if output_transform is not None:
            out = output_transform(out)
        if isinstance(out, (list, tuple)):
            for o in out:
                _build_graph(o.grad_fn)
        else:
            _build_graph(out.grad_fn)
        return module_to_node

    def _set_fc_index_transform(self, fc_node: Node):
        if fc_node.type != OPTYPE.LINEAR:
            return
        fc_in_features = fc_node.module.in_features  # the number of input features
        feature_channels = _get_in_node_out_channels(fc_node.inputs[0])  # the number of output channels of the input node
        stride = fc_in_features // feature_channels
        if stride > 1:
            # which means the input node of `fc_node` is a conv node
            for in_node in fc_node.inputs:
                for dep in fc_node.dependencies:
                    if dep.broken_node == in_node:
                        dep.index_transform = _FlattenIndexTransform(stride=stride, reverse=True)
                for dep in in_node.dependencies:
                    if dep.broken_node == fc_node:
                        dep.index_transform = _FlattenIndexTransform(stride=stride, reverse=False)

    def _set_concat_index_transform(self, cat_node: Node):
        if cat_node.type != OPTYPE.CONCAT:
            return
        
        chs = []
        for node in cat_node.inputs:
            chs.append(_get_in_node_out_channels(node))

        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1]+ch)
        cat_node.module.offsets = offsets
        for i, in_node in enumerate(cat_node.inputs):
            for dep in cat_node.dependencies:
                if dep.broken_node == in_node:
                    dep.index_transform = _ConcatIndexTransform(offset=offsets[i:i+2], reverse=True)

            for dep in in_node.dependencies:
                if dep.broken_node == cat_node:
                    dep.index_transform = _ConcatIndexTransform(offset=offsets[i:i+2], reverse=False)

    def _set_split_index_transform(self, split_node: Node):
        if split_node.type != OPTYPE.SPLIT:
            return
        
        chs = []
        for node in split_node.outputs:
            chs.append(_get_out_node_in_channels(node))

        offsets = [0]
        for ch in chs:
            offsets.append(offsets[-1]+ch)
        split_node.module.offsets = offsets
        for i, out_node in enumerate(split_node.outputs):
            for dep in split_node.dependencies:
                if dep.broken_node == out_node:
                    dep.index_transform = _SplitIndexTransform(offset=offsets[i:i+2], reverse=False)

            for dep in out_node.dependencies:
                if dep.broken_node == split_node:
                    dep.index_transform = _SplitIndexTransform(offset=offsets[i:i+2], reverse=True)

def _get_in_node_out_channels(node):
    # get the number of output channel of the given node or its input node
    ch = _get_node_out_channel(node)
    if ch is None:
        ch = 0
        for in_node in node.inputs:
            if node.type == OPTYPE.CONCAT:
                ch += _get_in_node_out_channels(in_node)
            else:
                ch = _get_in_node_out_channels(in_node)
    return ch

def _get_out_node_in_channels(node):
    # get the number of input channel of the given node or its output node
    ch = _get_node_in_channel(node)
    if ch is None:
        ch = 0
        for out_node in node.outputs:
            if node.type == OPTYPE.SPLIT:
                ch += _get_out_node_in_channels(out_node)
            else:
                ch = _get_out_node_in_channels(out_node)
    return ch

#### 相对路径问题
1. 首先打开左侧的“运行和调试”一栏，点击“创建launch.json文件”会在工作目录中的`.vscode`文件夹里创建`launch.json`文件，或直接人为创建也可
2. 按照下述代码修改`launch.json`文件：
    ```json
    {
        // 使用 IntelliSense 了解相关属性。 
        // 悬停以查看现有属性的描述。
        // 欲了解更多信息，请访问: https://go.microsoft.com/fwlink/?linkid=830387
        "version": "0.2.0",
        "configurations": [
            {
                "name": "Python: 当前文件",
                "type": "python",
                "request": "launch",
                "program": "${file}",
                "console": "integratedTerminal",
                "cwd": "${fileDirname}",
                "justMyCode": true
            }
        ]
    }
    ```

#### 找不到 git
1. 点击左下角按钮开用户设置
2. 在设置页面搜索栏中输入 "git.path" 字样
3. 点击在`setting.json`中编辑
4. 在`git.path`后加入`"D:/ProgramFiles/Git/bin/git.exe"`即可，最后该文件部分内容如下所示：
```json
    {
        "workbench.iconTheme": "vscode-icons",
        "workbench.colorTheme": "monokai-charcoal",
        "window.zoomLevel": -1,
        "security.workspace.trust.untrustedFiles": "open",
        "git.path": "D:/ProgramFiles/Git/bin/git.exe"
    }
```

#### 设置 vscode 自动保存
1. 点击左下角按钮开用户设置
2. 在设置页面搜索栏中输入 "autosave" 字样
3. 在`Files: Auto Save`一栏的下拉菜单中选择`afterDelay`即可
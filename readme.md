###多因子回测系统
请开发前详细阅读此文件

1）运行前准备
    运行本系统需要 Python 3.7及以上版本

    运行本系统需要安装以下python库：
        Django
        numpy
        pandas
        pymongo
        tqdm
        **可直接使用 pip install {库名称} 安装，建议但不强制使用conda或者env等虚拟环境方式运行

2）运行本系统
    运行回测系统前需在根目录下准备一个或多个以下文件：
        MonthFactor.csv（若需要测试周频数据）
        rolling_over_factor.csv（若需要按照因子历史表现更改分组）
    
    需在本文件夹根目录下运行 
        python manage.py runserver
        或
        py manage.py runserver

    运行后结果应当如下
        Watching for file changes with StatReloader
        Performing system checks...

        System check identified no issues (0 silenced).
        XXX XX, 2023 - XX:XX:XX
        Django version 4.2.3, using settings 'factor_analysis.settings'
        Starting development server at http://127.0.0.1:8000/
        Quit the server with CONTROL-C.
        *如果第二大段缺失，大概率原因是无法访问服务器

    - 若运行输出无异常，则可用任意浏览器在所显示的地址（默认http://127.0.0.1:8000/app）访问本系统
    - 若希望局域网内所有设备均可使用，运行 python/py manage.py runserver 0.0.0.0:8000后可通过正在运行程序设备的ip访问（http://设备ip:8000/app）

3）本系统功能 & 调试
    
    本系统提供周、月频率的单、多因子模型回测功能，并能够在回测的基础上进一步优化因子分数、因子权重、选股方式等
    
    进入本系统页面后可以看到多种选项
        *若一个选项标注写有默认值，则可以直接留空
        *无默认值的选项会标红提示

    点击“开始分析”按钮并等按钮变黄色后即代表程序正在运行，按钮变蓝之后即代表运行结束
        *运行结束后可点击 “回测数据、相关性矩阵下载” 下载数据，其中包括以下数据：
            - alpha_indicator.csv （每组alpha、回撤、calmar）
            - factor_corr.csv（所有所选因子之间相关性矩阵）
            - grouped_month_ret.csv（所有组合每日累计净值）
            - IC_result.csv（所选因子组合单月、累计IC值）
            - indicator.csv（每组收益、回撤、夏普率）

    本系统的多个功能均可以任意组合方式共同使用

    现有优化功能：
        - 个股权重根据指数内行业中性化
        - 根据历史IC智能选择使用哪些因子
        - 根据历史IC智能选择因子权重
        - 根据历史因子calmar表现选择最优组（因为有概率不是第0组）

4）开发注意事项
    
    1. 相关文件目录树
        .     
        |-- ...                   
        |-- readme.md（本文件）
        |-- MonthFactor.csv（必要输入）
        |-- rolling_over_factor.csv（必要输入）
        |-- test.py（用于测试）
        |-- requirements.txt（运行所需的库）
        |-- factorCombination.py（使用模型运行单因子测试的函数）
        |-- csv_result
        | |-- 本文件夹用于存储分析结束后供下载的各种数据
        |-- app
        | |-- ...
        | |-- templates
        | | |-- analysis_frontend.html（前端显示代码，使用html/css/js/Vue）
        | |-- factor
        | | |-- ...
        | | |-- factorModel.py（后端主要代码） 
        | | |-- PortfolioOptimization.py（用于行业中性化的辅助代码） 
    
        其他文件正常情况无需改变
        *除factorModel.py外大部份文件都应当避免改动

    2. 相关注意事项
        使用：
            - 前端使用了Vue框架进行渲染，所以需要联网才能使用
            - 新设备第一次打开前端网站较慢（<= 30s）为正常情况，无异常。
                若超过了30s依然无法打开，检查网络
            - 前端没有设置任何错误处理功能，所以如果长时间（> 5min）不出结果，需检查程序console是否报错。若有报错需重启程序+刷新网站

        开发：
            - 本程序直接接入mongoDB服务器，所以需在内网运行
            - 因对处理速度优化等原因，本程序严重依赖数据库内股票名称、因子名称在不同数据库内顺序的一致性。若顺序发生了改动则需要对代码索引方式进行改动
                *具体索引方式详见factorModel.py函数内注释
    
    3. 重要文件MD5值如下，请后续迭代重大版本时同时更改此值、时间
        factorModel.py 2023/08/25 15:19 UTC+08:00
            b93189f0409700f44cbeae1dda7529c9

        PortfolioOptimization.py 2023/08/25 15:19 UTC+08:00
            7df7d54701ce9d71d6a631cc6e9fc463

        analysis_frontend.html 2023/08/25 15:19 UTC+08:00
            1d322e757af67ea7faffc787567e1b6b

上次更新：2023/08/25
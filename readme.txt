1. 编译程序 python setup.py build_ext --inplace (修改63行arch 架构 code =xx arch=xx)
2. 修改配置文件 configs/*.ini
3. 修改 run.py 里的ini file选项 （39行）
4. 启动程序 python run.py
5. plot/plot.py （408，409行）修改mat 目录 和 输出目录，运行python plot.py 看结果


#### 可以使用一下命令创建一个可以跑的环境 （python3.12）
conda env create -f environment.yml

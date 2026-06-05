from __future__ import annotations


ZH_GUIDE = """FastBF 防忘指南

1. 安装和自检
   make install
   fastbf doctor

2. 无真实数据时，先跑内置安装烟测
   cd example
   python make_synthetic_sac.py
   fastbf doctor config.ini
   fastbf prepare config.ini
   fastbf checkmem config.ini --safety 0.8
   fastbf run config.ini --serial --clean-output
   python plot_beamforming.py

   目标图:
   example/workspace/plots/capon_bartlett_db.png

3. 跑自己的真实数据
   fastbf init -o my.ini
   # 修改 [seisarray.*], [time_filter], [window], [bands], [device], [output]
   fastbf doctor my.ini
   fastbf prepare my.ini
   fastbf checkmem my.ini --safety 0.8
   fastbf run my.ini --dry-run --serial --limit 1
   fastbf run my.ini

4. 记住 workspace 输出
   [output].dir/
     config.snapshot.ini
     nsl_catalog.tsv
     saclist/*.tsv
     native/
     bf/
     log/

5. 如果忘了 synthetic、真实数据和本地测试的区别
   example/ 只做安装 smoke test，不用于科学判断。
   真实数据请用自己的 ini 和 SAC 路径。
   testing/ 若存在，只是作者本机 ignored 沙盒，不随仓库发布。

更多细节:
   README.md
   docs/USAGE_GUARDRAILS.md
"""


EN_GUIDE = """FastBF usage reminder

1. Install and self-check
   make install
   fastbf doctor

2. Without real data, run the bundled installation smoke test
   cd example
   python make_synthetic_sac.py
   fastbf doctor config.ini
   fastbf prepare config.ini
   fastbf checkmem config.ini --safety 0.8
   fastbf run config.ini --serial --clean-output
   python plot_beamforming.py

   Expected plot:
   example/workspace/plots/capon_bartlett_db.png

3. Run real data
   fastbf init -o my.ini
   # Edit [seisarray.*], [time_filter], [window], [bands], [device], [output]
   fastbf doctor my.ini
   fastbf prepare my.ini
   fastbf checkmem my.ini --safety 0.8
   fastbf run my.ini --dry-run --serial --limit 1
   fastbf run my.ini

4. Workspace outputs
   [output].dir/
     config.snapshot.ini
     nsl_catalog.tsv
     saclist/*.tsv
     native/
     bf/
     log/

5. Synthetic, real data, and local tests
   example/ is only an installation smoke test, not a scientific reference.
   Real data should use your own config and SAC paths.
   testing/, if present, is an ignored author-local sandbox and is not shipped.

More details:
   README.en.md
   docs/USAGE_GUARDRAILS.md
"""


def guide_text(lang: str = "zh") -> str:
    if lang == "en":
        return EN_GUIDE
    return ZH_GUIDE

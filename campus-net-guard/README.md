# Campus Net Guard

校园网自动登录脚本（已去除账号密码）。

## 使用
1. 编辑 `campus-net-guard.json`，填入你自己的账号密码
2. 测试：

```powershell
python campus_net_guard.py --once
```

3. 常驻运行：

```powershell
python campus_net_guard.py
```

## 说明
- 本仓库中的配置文件是模板，不含真实账号密码
- 日志会写到 `workspace/logs/`

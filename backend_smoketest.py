#!/usr/bin/env python3
# 宠 IP 工坊 · 后端连通性冒烟测试（纯标准库，无需安装依赖）
# 用法：
#   1) 先启动后端：python main.py   （监听 http://localhost:8000）
#   2) 另开一个终端运行：python backend_smoketest.py
import json, urllib.request, urllib.error

BASE = "http://localhost:8000"

def call(method, path, body=None, is_json=True):
    url = BASE + path
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read(2000)
    except urllib.error.HTTPError as e:
        return e.code, e.read(500)
    except Exception as e:
        return None, str(e).encode()

# (说明, 方法, 路径, 请求体)  —— 只读探测，不做破坏性写入
PROBES = [
    ("健康检查",          "GET",  "/api/health",          None),
    ("素材列表",          "GET",  "/api/materials",       None),
    ("IP 人设档案",       "GET",  "/api/ip-profile",      None),
    ("设备列表",          "GET",  "/api/devices",         None),
    ("统计数据",          "GET",  "/api/stats",           None),
    ("生活小传(列表)",     "GET",  "/api/generate/story",  None),
    ("IP 图片(列表)",      "GET",  "/api/generate/ip",     None),
    ("3D 模型(列表)",      "GET",  "/api/generate/3d",     None),
    ("生成小传(写)",       "POST", "/api/generate/story",  {"pet_name":"喵小懒","personality":["慵懒"],"style":"治愈"}),
]

print(f"探测目标：{BASE}\n" + "-"*54)
ok = fail = 0
for name, method, path, body in PROBES:
    status, payload = call(method, path, body)
    good = status is not None and 200 <= status < 300
    ok += good; fail += (not good)
    tag = "✅" if good else "❌"
    extra = "" if good else f"  -> {payload[:120]!r}"
    print(f"{tag} {method:4} {path:28} [{status}] {name}{extra}")
print("-"*54)
print(f"通过 {ok} / 失败 {fail}")
print("提示：若全部 ❌ 且为连接错误，说明后端未启动或端口不是 8000。")

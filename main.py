"""
宠 IP 工坊 - 基于 AI 智能宠物监护摄像头的动态 IP 创作系统 后端 API
技术栈: FastAPI + Python 3.10+
"""

import os
import json
import shutil
import hashlib
import base64
import aiofiles
from datetime import datetime
from typing import Optional, Any

from dotenv import load_dotenv
from openai import OpenAI
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pathlib import Path

import httpx
import uuid
import asyncio

# 加载 .env 环境变量
load_dotenv()

# ModelScope 魔搭 FLUX 客户端（兼容 OpenAI 接口，关闭代理）
try:
    modelscope_client = OpenAI(
        api_key=os.getenv("MODELSCOPE_API_KEY"),
        base_url=os.getenv("MODELSCOPE_BASE_URL", "https://api-inference.modelscope.cn/v1"),
        http_client=httpx.Client(proxy=None),
    )
    print(f"[ModelScope] base_url={modelscope_client.base_url} key={modelscope_client.api_key[:20]}...")
except Exception as e:
    print(f"[ModelScope] 初始化失败: {e}")
    modelscope_client = None


# ============================================================
# 应用初始化
# ============================================================
app = FastAPI(title="宠 IP 工坊 API", version="1.1.0")

# 跨域配置 - 允许所有来源（开发环境）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Tripo3D 配置
# ============================================================
TRIPO_API_KEY = os.getenv("TRIPO_API_KEY", "")
TRIPO_BASE_URL = "https://api.tripo3d.ai/v2/openapi"
TRIPO_MODEL_VERSION = os.getenv("TRIPO_MODEL_VERSION", "v3.1-20260211")
TRIPO_GEOMETRY_QUALITY = os.getenv("TRIPO_GEOMETRY_QUALITY", "standard")
TRIPO_ENABLE_TEXTURE = os.getenv("TRIPO_ENABLE_TEXTURE", "false").lower() == "true"
TRIPO_AUTO_SIZE = os.getenv("TRIPO_AUTO_SIZE", "true").lower() == "true"
TRIPO_ENABLE_AUTOFIX = os.getenv("TRIPO_ENABLE_AUTOFIX", "true").lower() == "true"
TRIPO_TASK_TIMEOUT_SECONDS = int(os.getenv("TRIPO_TASK_TIMEOUT_SECONDS", "900"))
TRIPO_POLL_INTERVAL_SECONDS = float(os.getenv("TRIPO_POLL_INTERVAL_SECONDS", "5"))
DEFAULT_3D_SOURCE_DIR = Path(os.getenv("DEFAULT_3D_SOURCE_DIR", str(Path(__file__).parent.resolve() / "Milky酱" / "Milky酱照片")))
DEFAULT_3D_SOURCE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

HAPPYHORSE_API_KEY = os.getenv("HAPPYHORSE_API_KEY", "")
HAPPYHORSE_BASE_URL = os.getenv("HAPPYHORSE_BASE_URL", "https://ws-z2lqx5wip4a51hs5.ap-southeast-1.maas.aliyuncs.com/api/v1").rstrip("/")
HAPPYHORSE_MODEL = os.getenv("HAPPYHORSE_MODEL", "happyhorse-1.0-i2v")
HAPPYHORSE_VIDEO_RATIO = os.getenv("HAPPYHORSE_VIDEO_RATIO", "1280:720")
HAPPYHORSE_VIDEO_SIZE = os.getenv("HAPPYHORSE_VIDEO_SIZE", HAPPYHORSE_VIDEO_RATIO.replace(":", "*"))
HAPPYHORSE_VIDEO_DURATION = int(os.getenv("HAPPYHORSE_VIDEO_DURATION", "5"))
HAPPYHORSE_TASK_TIMEOUT_SECONDS = int(os.getenv("HAPPYHORSE_TASK_TIMEOUT_SECONDS", "900"))
HAPPYHORSE_POLL_INTERVAL_SECONDS = float(os.getenv("HAPPYHORSE_POLL_INTERVAL_SECONDS", "5"))
DEFAULT_IP_IMAGE_DIR = Path(os.getenv("DEFAULT_IP_IMAGE_DIR", str(Path(__file__).parent.resolve() / "Milky酱" / "IP 形象")))
DEFAULT_IP_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# ============================================================
# 目录结构
# ============================================================
BASE_DIR = Path(__file__).parent.resolve()
DIRS = {
    "upload": BASE_DIR / "upload",        # 原始素材上传目录
    "gen_ip": BASE_DIR / "gen_ip",        # AI 生成的 IP 形象
    "gen_comic": BASE_DIR / "gen_comic",  # AI 生成的四格漫画
    "gen_video": BASE_DIR / "gen_video",  # AI 生成的精华短片
    "gen_3d": BASE_DIR / "gen-3d",        # AI 生成的 3D 模型文件
}
DATA_FILE = BASE_DIR / "data.json"         # 持久化数据文件

# 确保所有目录存在
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# 异步任务存储（用于 Tripo3D / HappyHorse 等长时间任务）
_tasks: dict[str, dict] = {}

# 静态文件路由（用于前端直接访问图片/文件）
for mount_path, dir_path in [
    ("/static/upload", DIRS["upload"]),
    ("/static/gen_ip", DIRS["gen_ip"]),
    ("/static/gen_comic", DIRS["gen_comic"]),
    ("/static/gen_video", DIRS["gen_video"]),
    ("/static/gen_3d", DIRS["gen_3d"]),
]:
    app.mount(mount_path, StaticFiles(directory=str(dir_path)), name=mount_path.strip("/").replace("/", "_"))

# 通用静态文件目录（存放 监控模拟.jpg 等文件）
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

# ============================================================
# 前端页面托管（让后端直接提供 index.html）
# ============================================================
INDEX_HTML = BASE_DIR / "index.html"

if INDEX_HTML.exists():
    @app.get("/", response_class=FileResponse, include_in_schema=False)
    async def serve_frontend():
        return FileResponse(str(INDEX_HTML))

    @app.exception_handler(404)
    async def not_found_handler(request, exc):
        # 非 /api 开头的路径返回 index.html（支持前端路由）
        if not request.url.path.startswith("/api"):
            if INDEX_HTML.exists():
                return FileResponse(str(INDEX_HTML))
        return JSONResponse(status_code=404, content={"detail": "Not found"})


# ============================================================
# Pydantic 数据模型
# ============================================================
class DeviceBase(BaseModel):
    name: str
    group: str = "默认"


class GenerateIPRequest(BaseModel):
    prompt: str = "可爱的宠物猫IP形象，Q版风格，治愈系"
    style: str = "Q版萌系"
    material_ids: list[int] = []


class GenerateComicRequest(BaseModel):
    prompt: str = "宠物日常趣事四格漫画"
    style: str = "日系漫画"
    material_ids: list[int] = []


class GenerateStoryRequest(BaseModel):
    prompt: str = ""
    pet_name: str = "我的宠物"
    personality: list[str] = ["活泼", "可爱"]
    style: str = "温暖治愈"


class GenerateVideoRequest(BaseModel):
    prompt: str = ""
    duration: int = HAPPYHORSE_VIDEO_DURATION
    ratio: str = HAPPYHORSE_VIDEO_RATIO


class Generate3DRequest(BaseModel):
    prompt: str = "可爱的宠物Q版手办"
    pose: str = "站立"
    color_scheme: str = "#FF8C8C"
    accessory: str = "无"
    size: str = "10cm"
    material_ids: list[int] = []


class IPProfileUpdate(BaseModel):
    name: Optional[str] = None
    style: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    avatar_material_id: Optional[int] = None


# ============================================================
# JSON 持久化存储（替代内存数据库，重启不丢数据）
# ============================================================
class SimpleDB:
    """JSON 文件持久化存储，所有写操作自动保存到 data.json"""

    def __init__(self):
        self._materials = []
        self._ip_images = []
        self._comics = []
        self._stories = []
        self._videos = []
        self._models_3d = []
        self._ip_profile = {
            "name": "喵小懒",
            "style": "治愈系 / 萌趣 / 日常生活",
            "bio": "一只热爱阳光和美食的橘猫，白天喜欢趴在窗台看风景，晚上变身夜行探险家。虽然看起来懒洋洋的，但对美食有着敏锐的嗅觉和执着。",
            "avatar_url": "",
            "avatar_material_id": None,
            "updated_at": "",
        }
        self._devices = [
            {"id": 1, "name": "客厅摄像头-01", "group": "客厅", "status": "online", "last_sync": "2分钟前", "ip": "192.168.1.101"},
            {"id": 2, "name": "卧室摄像头-01", "group": "卧室", "status": "online", "last_sync": "5分钟前", "ip": "192.168.1.102"},
            {"id": 3, "name": "阳台摄像头-01", "group": "阳台", "status": "offline", "last_sync": "1小时前", "ip": "192.168.1.103"},
        ]
        self._next_id = 1
        self._load()

    # ---- 持久化 ----
    def _load(self):
        """从 JSON 文件加载数据"""
        if DATA_FILE.exists():
            try:
                data = json.loads(DATA_FILE.read_text(encoding="utf-8"))
                self._materials = data.get("materials", [])
                self._ip_images = data.get("ip_images", [])
                self._comics = data.get("comics", [])
                self._stories = data.get("stories", [])
                self._videos = data.get("videos", [])
                self._models_3d = data.get("models_3d", [])
                self._ip_profile.update(data.get("ip_profile", {}))
                self._devices = data.get("devices", self._devices)
                self._next_id = data.get("next_id", 1)
                print(f"   📄 已加载 {DATA_FILE.name}（素材{len(self._materials)}条，IP形象{len(self._ip_images)}条，漫画{len(self._comics)}条，故事{len(self._stories)}条，3D模型{len(self._models_3d)}条）")
            except Exception as e:
                print(f"   ⚠️  data.json 读取失败，使用默认数据: {e}")

    def _save(self):
        """将数据写入 JSON 文件"""
        data = {
            "materials": self._materials,
            "ip_images": self._ip_images,
            "comics": self._comics,
            "stories": self._stories,
            "videos": self._videos,
            "models_3d": self._models_3d,
            "ip_profile": self._ip_profile,
            "devices": self._devices,
            "next_id": self._next_id,
        }
        DATA_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- 素材 CRUD ----
    def add_material(self, filename: str, original_name: str, file_size: int, file_type: str, behavior: str = "未知", emotion: str = "未知"):
        # 去重：同名同大小的文件不重复添加
        for m in self._materials:
            if m["original_name"] == original_name and m["file_size"] == file_size:
                return m
        item = {
            "id": self._next_id,
            "filename": filename,
            "original_name": original_name,
            "url": f"/static/upload/{filename}",
            "file_size": file_size,
            "file_type": file_type,
            "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "behavior": behavior,
            "emotion": emotion,
            "scene": "手动上传",
            "tags": ["手动上传"],
            "favorite": False,
            "type": "image" if file_type.startswith("image") else "video",
        }
        self._next_id += 1
        self._materials.insert(0, item)
        self._save()
        return item

    def get_materials(self):
        return self._materials

    def get_material(self, material_id: int):
        for m in self._materials:
            if m["id"] == material_id:
                return m
        return None

    def delete_material(self, material_id: int):
        for i, m in enumerate(self._materials):
            if m["id"] == material_id:
                filepath = DIRS["upload"] / m["filename"]
                if filepath.exists():
                    filepath.unlink()
                removed = self._materials.pop(i)
                self._save()
                return removed
        return None

    def toggle_favorite(self, material_id: int):
        m = self.get_material(material_id)
        if m:
            m["favorite"] = not m["favorite"]
            self._save()
            return m
        return None

    def update_material_tags(self, material_id: int, tags: list[str]):
        m = self.get_material(material_id)
        if m:
            existing = set(m["tags"])
            for t in tags:
                existing.add(t)
            m["tags"] = list(existing)
            self._save()
            return m
        return None

    # ---- IP 形象 ----
    def add_ip_image(self, filename: str, prompt: str, style: str):
        item = {
            "id": self._next_id,
            "filename": filename,
            "url": f"/static/gen_ip/{filename}",
            "prompt": prompt,
            "style": style,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._next_id += 1
        self._ip_images.insert(0, item)
        self._save()
        return item

    def get_ip_images(self):
        return self._ip_images

    # ---- 漫画 ----
    def add_comic(self, filenames: list[str], prompt: str):
        item = {
            "id": self._next_id,
            "frames": [f"/static/gen_comic/{f}" for f in filenames],
            "frame_files": filenames,
            "prompt": prompt,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._next_id += 1
        self._comics.insert(0, item)
        self._save()
        return item

    def get_comics(self):
        return self._comics

    # ---- 故事 ----
    def add_story(self, title: str, content: str, pet_name: str):
        item = {
            "id": self._next_id,
            "title": title,
            "content": content,
            "pet_name": pet_name,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._next_id += 1
        self._stories.insert(0, item)
        self._save()
        return item

    def get_stories(self):
        return self._stories

    # ---- 精华短片 ----
    def add_video(self, filename: str, prompt: str, source_image: str, provider_task_id: str, provider: str = "HappyHorse"):
        item = {
            "id": self._next_id,
            "filename": filename,
            "url": f"/static/gen_video/{filename}",
            "prompt": prompt,
            "source_image": source_image,
            "provider": provider,
            "provider_task_id": provider_task_id,
            "runway_task_id": provider_task_id,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._next_id += 1
        self._videos.insert(0, item)
        self._save()
        return item

    def get_videos(self):
        return self._videos

    # ---- 3D 模型 ----
    def add_3d_model(self, filename: str, prompt: str, pose: str, format_type: str, extra: Optional[dict] = None):
        extra = extra or {}
        stl_filename = extra.get("stl_filename") or (filename if str(filename).lower().endswith(".stl") else "")
        mf3_filename = extra.get("3mf_filename") or (filename if str(filename).lower().endswith(".3mf") else "")
        glb_filename = extra.get("glb_filename") or (filename if str(filename).lower().endswith(".glb") else "")
        item = {
            "id": self._next_id,
            "filename": filename,
            "url": f"/static/gen_3d/{filename}" if filename else "",
            "stl_filename": stl_filename,
            "3mf_filename": mf3_filename,
            "glb_filename": glb_filename,
            "stl_url": f"/static/gen_3d/{stl_filename}" if stl_filename else "",
            "3mf_url": f"/static/gen_3d/{mf3_filename}" if mf3_filename else "",
            "glb_url": f"/static/gen_3d/{glb_filename}" if glb_filename else "",
            "prompt": prompt,
            "pose": pose,
            "format": format_type,
            "tripo_task_id": extra.get("tripo_task_id", ""),
            "source_material_id": extra.get("source_material_id"),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        self._next_id += 1
        self._models_3d.insert(0, item)
        self._save()
        return item

    def get_3d_models(self):
        return self._models_3d

    # ---- IP 档案 ----
    def get_ip_profile(self):
        return self._ip_profile

    def update_ip_profile(self, data: dict):
        for field in ("name", "style", "bio", "avatar_url"):
            if field in data and data[field] is not None:
                self._ip_profile[field] = data[field]
        if "avatar_material_id" in data:
            self._ip_profile["avatar_material_id"] = data["avatar_material_id"]
        self._ip_profile["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        self._save()
        return self._ip_profile

    # ---- 设备 ----
    def get_devices(self):
        return self._devices

    def add_device(self, name: str, group: str):
        item = {
            "id": self._next_id,
            "name": name,
            "group": group,
            "status": "online",
            "last_sync": "刚刚",
            "ip": f"192.168.1.{100 + self._next_id}",
        }
        self._next_id += 1
        self._devices.append(item)
        self._save()
        return item

    def update_device(self, device_id: int, data: dict):
        for d in self._devices:
            if d["id"] == device_id:
                d.update(data)
                self._save()
                return d
        return None


db = SimpleDB()


# ============================================================
# 工具函数
# ============================================================
def generate_unique_filename(original_name: str) -> str:
    """生成唯一文件名，保留原始扩展名"""
    ext = os.path.splitext(original_name)[1] or ".png"
    unique = hashlib.md5(f"{uuid.uuid4()}_{datetime.now().timestamp()}".encode()).hexdigest()[:16]
    return f"{unique}{ext}"


# ============================================================
# 动态素材库 · Milky 相册种子数据（首次启动导入，使列表由后端统一管理）
# ============================================================
DEMO_MATERIALS_DIR = BASE_DIR / "Milky酱" / "demo_materials"
DEMO_MATERIALS_META = [
    ("1.jpg",  "思考",   "安静"),
    ("2.jpg",  "撒娇",     "开心"),
    ("4.jpg",  "晒太阳",     "兴奋"),
    ("5.jpg",  "晒太阳",     "安静"),
    ("6.jpeg", "发呆",     "好奇"),
    ("7.jpg",  "拜年",     "开心"),
    ("8.jpg",  "迷倒路人", "开心"),
    ("9.jpg",  "思考",     "郁闷"),
    ("10.jpg", "睡觉",     "安静"),
]
DEMO_MATERIAL_SCENE = "Milky相册"


def seed_demo_materials():
    """启动时把 demo 照片复制到 upload/ 并重建素材库记录（适配 Railway 临时文件系统，每次重启自动恢复）。"""
    if not DEMO_MATERIALS_DIR.exists():
        return

    # 清理旧 demo 记录（Railway 重启后 data.json 可能残留指向已丢失文件的旧记录）
    db._materials[:] = [m for m in db._materials if m.get("scene") != DEMO_MATERIAL_SCENE]

    seeded = 0
    for fname, behavior, emotion in DEMO_MATERIALS_META:
        src = DEMO_MATERIALS_DIR / fname
        if not src.exists():
            continue
        target_name = generate_unique_filename(fname)
        target = DIRS["upload"] / target_name
        try:
            shutil.copy2(src, target)
        except Exception as e:
            print(f"[SEED] 复制失败 {src}: {e}")
            continue
        ext = src.suffix.lower()
        file_type = "image/jpeg" if ext in (".jpg", ".jpeg") else "image/png"
        item = db.add_material(
            filename=target_name,
            original_name=fname,
            file_size=target.stat().st_size,
            file_type=file_type,
            behavior=behavior,
            emotion=emotion,
        )
        item["scene"] = DEMO_MATERIAL_SCENE
        item["tags"] = list(dict.fromkeys(item.get("tags", []) + ["Milky", "相册"]))
        seeded += 1
    if seeded:
        db._save()
        print(f"[SEED] 已导入 {seeded} 张 Milky 相册素材")


def import_default_3d_source_material() -> Optional[int]:
    """从 Milky 照片目录导入一张图片作为 3D 生成默认素材。"""
    source_dir = DEFAULT_3D_SOURCE_DIR
    if not source_dir.exists():
        print(f"[3D] 默认素材目录不存在: {source_dir}")
        return None

    existing = [
        m for m in db.get_materials()
        if m.get("type") == "image" and m.get("source_path", "").startswith(str(source_dir))
    ]
    if existing:
        return existing[0]["id"]

    candidates = sorted(
        p for p in source_dir.iterdir()
        if p.is_file() and p.suffix.lower() in DEFAULT_3D_SOURCE_EXTENSIONS
    )
    if not candidates:
        print(f"[3D] 默认素材目录没有可用图片: {source_dir}")
        return None

    source_path = candidates[0]
    filename = generate_unique_filename(source_path.name)
    target_path = DIRS["upload"] / filename
    shutil.copy2(source_path, target_path)
    file_type = _content_type_for_image(source_path) if "_content_type_for_image" in globals() else "image/jpeg"
    item = db.add_material(
        filename=filename,
        original_name=source_path.name,
        file_size=target_path.stat().st_size,
        file_type=file_type,
        behavior="3D素材",
        emotion="开心",
    )
    item["source_path"] = str(source_path)
    item["tags"] = list(dict.fromkeys(item.get("tags", []) + ["Milky", "3D素材"]))
    db._save()
    print(f"[3D] 已导入默认素材: {source_path} -> material_id={item['id']}")
    return item["id"]


def generate_placeholder_image(target_path: Path, width: int = 512, height: int = 512, text: str = "Placeholder"):
    """
    生成占位图片 SVG 文件（不依赖 PIL），
    当 AI API 尚未接入时，模拟生成结果以便前端调试。
    """
    svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
  <rect width="100%" height="100%" fill="#252542"/>
  <rect x="10%" y="10%" width="80%" height="80%" rx="20" fill="#3D3D6B"/>
  <text x="50%" y="45%" font-family="Arial" font-size="72" fill="#FF6B9D" text-anchor="middle">🐾</text>
  <text x="50%" y="55%" font-family="Arial" font-size="16" fill="#A0A0B8" text-anchor="middle">{text}</text>
</svg>"""
    target_path.write_text(svg_content, encoding="utf-8")


# ============================================================
# 硅基流动（SiliconFlow）AI API：看图写小传 + 图生图
# 从 main(1).py 复刻：
#   call_siliconflow_vl_api  —— Qwen3-VL 看宠物图生成人物小传
#   call_siliconflow_api     —— 图生图生成图片（默认 FLUX.1-dev，可配 SILICONFLOW_IMAGE_MODEL）
# ============================================================

# 硅基流动 API Key（复刻自 main(1).py）
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "sk-xxqkhuuywtdvelcneecidzpynbqrofrxvoejyuhxzwuhtsaw")
SILICONFLOW_IMAGE_MODEL = os.getenv("SILICONFLOW_IMAGE_MODEL", "Tongyi-MAI/Z-Image-Turbo")
SILICONFLOW_VL_MODEL = os.getenv("SILICONFLOW_VL_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# Ark/火山引擎 Seedream
ARK_API_KEY = os.getenv("ARK_API_KEY", "ark-bb8b2c5c-94f9-45a7-b8b1-9db6187d8721-4b27d")
ARK_BASE_URL = os.getenv("ARK_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3")
ARK_MODEL = os.getenv("ARK_MODEL", "doubao-seedream-4-5-251128")
try:
    ark_client = OpenAI(api_key=ARK_API_KEY, base_url=ARK_BASE_URL)
    print(f"[Ark] 初始化完成")
except Exception as e:
    print(f"[Ark] 初始化失败: {e}")
    ark_client = None


async def call_siliconflow_vl_api(
    prompt: str, image_path: Path, style: str,
    style_ref_path: Optional[Path] = None,
    instruction_override: Optional[str] = None
) -> str:
    """
    硅基流动 Qwen3-VL-8B-Instruct 多模态模型。
    如果提供 style_ref_path，同时看风格参考图 + 素材图，生成综合描述（画风+角色）；
    否则只看素材图写角色小传。
    instruction_override 不为空时，直接用它作为文字指令（用于周边产品图等场景）。
    """
    api_key = SILICONFLOW_API_KEY.strip()

    content_items = []

    # 风格参考图
    if style_ref_path and style_ref_path.exists():
        with open(style_ref_path, "rb") as f:
            sty_b64 = base64.b64encode(f.read()).decode()
        sty_mime = "image/png" if style_ref_path.suffix.lower() == ".png" else "image/jpeg"
        content_items.append({
            "type": "image_url",
            "image_url": {"url": f"data:{sty_mime};base64,{sty_b64}"}
        })
        print(f"[Qwen3-VL] 风格参考图: {style_ref_path.name}")

    # 宠物素材图
    with open(image_path, "rb") as f:
        pet_b64 = base64.b64encode(f.read()).decode()
    pet_mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
    content_items.append({
        "type": "image_url",
        "image_url": {"url": f"data:{pet_mime};base64,{pet_b64}"}
    })
    print(f"[Qwen3-VL] 宠物参考图: {image_path.name}")

    # 文字指令
    if instruction_override:
        text_instruction = instruction_override
    elif style_ref_path and style_ref_path.exists():
        text_instruction = (
            "请仔细观察两张图片。\n"
            "第一张是风格参考图，请分析其画风特点（线条、配色、质感等）；\n"
            "第二张是宠物照片，请提取宠物的品种、外貌特征、性格特点。\n"
            "然后综合两者，写一段简短的卡通漫画生成提示词，要求：\n"
            "1) 包含宠物的姓名（根据外貌起名）、品种、外貌、性格；\n"
            "2) 包含一个简短的生活场景（不超过30字）；\n"
            "3) 在提示词开头明确描述应采用的画风（模仿风格参考图）；\n"
            "4) 整体不超过150字，不包含任何特殊符号。"
        )
    else:
        text_instruction = (
            f"请用{style}风格写一段简短的宠物角色描述，用于后续AI绘制卡通漫画。"
            "要求：1) 包含姓名（根据外貌起名）、品种、外貌特征、性格特点；"
            "2) 包含简短生活场景（不超过30字）；"
            "3) 整体不超过100字。"
        )
    if prompt:
        text_instruction += f"\n附加需求：{prompt}"
    content_items.append({"type": "text", "text": text_instruction})

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": SILICONFLOW_VL_MODEL,
        "messages": [{"role": "user", "content": content_items}],
        "max_tokens": 300
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                json=payload,
                headers=headers
            )
            resp.raise_for_status()
            data = resp.json()
        content = data["choices"][0]["message"]["content"].strip()
        print(f"  ✅ VL 生成结果: {content[:100]}...")
        return content
    except Exception as e:
        print(f"[ERROR] Qwen3-VL 调用失败: {e}")
        return f"一只可爱的宠物，{style}风格，在愉快地玩耍。"


async def call_siliconflow_api(
        prompt: str,
        style: str,
        ref_image: str,  # 素材参考图（base64 data URI）
        output_dir: Path,
        prefix: str = "comic",
        full_prompt_override: Optional[str] = None,       # 直接指定完整提示词（如周边产品图）
        negative_prompt_override: Optional[str] = None,
) -> list[str]:
    """
    硅基流动 (SiliconFlow) 图生图 API。
    默认使用 FLUX.1-dev 模型（可通过 SILICONFLOW_IMAGE_MODEL 环境变量切换）。
    传入素材图+文字指令生成新图。
    传入 full_prompt_override 时直接使用该提示词（用于周边产品图等非漫画场景）。
    """
    filenames = []
    output_dir.mkdir(parents=True, exist_ok=True)

    api_key = SILICONFLOW_API_KEY.strip()

    if full_prompt_override:
        full_prompt = full_prompt_override
    else:
        full_prompt = (
            f"{prompt}，{style}，治愈系软萌水彩插画，手绘质感，真实笔触，柔和暖色调，尽量优化得可爱一点"
            "毛绒一定要有蓬松感，线条柔和，画面干净简洁，纯白背景，无阴影，高细节，高画质"
            "严格禁止任何文字、字符、符号、字母出现在画面中"
        )
    negative_prompt = negative_prompt_override or (
        "低画质，模糊，变形，残缺，硬线条，高对比，写实照片，复杂背景，阴影过重，色彩杂乱，畸形"
        "水印，签名，多余文字，畸变，杂乱背景"
    )

    print("\n" + "=" * 60)
    print(f"[硅基流动] 开始图生图")
    print(f"  提示词: {full_prompt[:120]}...")
    print("=" * 60 + "\n")

    if not api_key:
        print("[ERROR] API Key 缺失，降级为 SVG 占位图")
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.svg"
        filepath = output_dir / filename
        generate_placeholder_image(filepath, 512, 512, f"{prompt[:20]}\n({style})")
        filenames.append(filename)
        return filenames

    # 文生图模式（无参考图，直接生成）
    if not ref_image:
        print("[硅基流动] 文生图模式（无参考图）")

    try:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": SILICONFLOW_IMAGE_MODEL,
            "prompt": full_prompt,
            "image": ref_image,
            "image_size": "1024x1024",
            "num_images": 1
        }
        if negative_prompt:
            payload["negative_prompt"] = negative_prompt

        print("[硅基流动] 调用 API...")
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                "https://api.siliconflow.cn/v1/images/generations",
                json=payload,
                headers=headers
            )
            print(f"[DEBUG] 状态码: {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()

        image_url = data["images"][0]["url"]

        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
        filepath = output_dir / filename
        async with httpx.AsyncClient(timeout=60) as c:
            img_resp = await c.get(image_url)
            img_resp.raise_for_status()

        with open(filepath, "wb") as f:
            f.write(img_resp.content)
        filenames.append(filename)
        print(f"  ✅ 下载完成: {filename}")

    except Exception as e:
        print(f"[ERROR] 图生图调用失败: {e}")
        print("[ERROR] 降级为 SVG 占位图")
        filename = f"{prefix}_{uuid.uuid4().hex[:8]}.svg"
        filepath = output_dir / filename
        generate_placeholder_image(filepath, 512, 512, f"{prompt[:20]}\n({style})")
        filenames.append(filename)

    return filenames

def _latest_material_image_path() -> Optional[Path]:
    """取素材库中最新的一张图片，作为看图/图生图的参考素材。"""
    for m in db.get_materials():
        if m.get("type") == "image":
            p = DIRS["upload"] / m["filename"]
            if p.exists():
                return p
    return None

def _extract_tripo_data(payload: dict) -> dict:
    if not isinstance(payload, dict):
        return {}
    data = payload.get("data")
    if isinstance(data, dict):
        return data
    return payload


def _find_first_key(obj: Any, keys: tuple[str, ...]) -> Any:
    if isinstance(obj, dict):
        for key in keys:
            if key in obj and obj[key]:
                return obj[key]
        for value in obj.values():
            found = _find_first_key(value, keys)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_first_key(value, keys)
            if found:
                return found
    return None


def _collect_urls(obj: Any) -> list[str]:
    urls = []
    if isinstance(obj, dict):
        for value in obj.values():
            urls.extend(_collect_urls(value))
    elif isinstance(obj, list):
        for value in obj:
            urls.extend(_collect_urls(value))
    elif isinstance(obj, str) and obj.startswith(("http://", "https://")):
        urls.append(obj)
    return urls


def _extension_from_url(url: str, fallback: str = ".glb") -> str:
    path = url.split("?", 1)[0].split("#", 1)[0]
    ext = os.path.splitext(path)[1].lower()
    if ext in {".stl", ".3mf", ".glb", ".obj", ".fbx", ".usdz", ".zip"}:
        return ext
    return fallback


def _content_type_for_image(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
    }.get(ext, "application/octet-stream")


async def _tripo_request(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> dict:
    url = f"{TRIPO_BASE_URL}{path}"
    try:
        resp = await client.request(method, url, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as exc:
        raise HTTPException(
            status_code=504,
            detail=f"无法连接 Tripo3D API（{TRIPO_BASE_URL}）。请检查网络、DNS 或代理后重试。原始错误: {exc}"
        )
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=f"Tripo3D API 请求失败({resp.status_code}): {payload}")
    if isinstance(payload, dict) and payload.get("code") not in (None, 0, 200, "0", "200"):
        raise HTTPException(status_code=502, detail=f"Tripo3D API 返回错误: {payload}")
    return payload


async def _upload_image_to_tripo(client: httpx.AsyncClient, image_path: Path) -> dict:
    headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}
    async with aiofiles.open(image_path, "rb") as f:
        content = await f.read()
    files = {"file": (image_path.name, content, _content_type_for_image(image_path))}
    payload = await _tripo_request(client, "POST", "/upload", headers=headers, files=files)
    data = _extract_tripo_data(payload)
    token = _find_first_key(data, ("image_token", "file_token", "token", "file_id", "id"))
    if not token:
        raise HTTPException(status_code=502, detail=f"Tripo3D 上传成功但未返回文件 token: {payload}")
    return {"token": token, "raw": data}


async def _create_tripo_task(client: httpx.AsyncClient, payload: dict) -> str:
    headers = {"Authorization": f"Bearer {TRIPO_API_KEY}", "Content-Type": "application/json"}
    response = await _tripo_request(client, "POST", "/task", headers=headers, json=payload)
    data = _extract_tripo_data(response)
    task_id = _find_first_key(data, ("task_id", "taskId", "id"))
    if not task_id:
        raise HTTPException(status_code=502, detail=f"Tripo3D 未返回 task_id: {response}")
    return str(task_id)


async def _poll_tripo_task(client: httpx.AsyncClient, task_id: str) -> dict:
    headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}
    deadline = asyncio.get_running_loop().time() + TRIPO_TASK_TIMEOUT_SECONDS
    last_payload = None
    while asyncio.get_running_loop().time() < deadline:
        payload = await _tripo_request(client, "GET", f"/task/{task_id}", headers=headers)
        data = _extract_tripo_data(payload)
        last_payload = data
        status = str(_find_first_key(data, ("status", "task_status", "state")) or "").lower()
        print(f"[Tripo3D] task={task_id} status={status or 'unknown'}")
        if status in {"success", "succeeded", "finished", "completed", "complete"}:
            return data
        if status in {"failed", "failure", "error", "cancelled", "canceled"}:
            raise HTTPException(status_code=502, detail=f"Tripo3D 任务失败: {data}")
        await asyncio.sleep(TRIPO_POLL_INTERVAL_SECONDS)
    raise HTTPException(status_code=504, detail=f"Tripo3D 任务超时: {last_payload}")


async def _download_model_urls(client: httpx.AsyncClient, task_id: str, task_data: dict, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {TRIPO_API_KEY}"}
    urls = []
    for url in _collect_urls(task_data):
        ext = _extension_from_url(url)
        if ext in {".stl", ".3mf", ".glb", ".obj", ".fbx", ".usdz", ".zip"}:
            urls.append(url)
    seen = set()
    downloaded = {}
    for index, url in enumerate(urls, start=1):
        if url in seen:
            continue
        seen.add(url)
        ext = _extension_from_url(url, ".glb")
        key = "3mf" if ext == ".3mf" else ext.lstrip(".")
        filename = f"tripo_{task_id}_{index}{ext}"
        path = output_dir / filename
        resp = await client.get(url, headers=headers, follow_redirects=True)
        if resp.status_code >= 400:
            resp = await client.get(url, follow_redirects=True)
        resp.raise_for_status()
        async with aiofiles.open(path, "wb") as f:
            await f.write(resp.content)
        downloaded.setdefault(key, filename)
        print(f"[Tripo3D] downloaded {key}: {filename}")
    return downloaded


async def call_tripo_image_to_3d(material_id: int, output_dir: Path) -> dict:
    if not TRIPO_API_KEY:
        raise HTTPException(status_code=500, detail="Tripo3D API Key 未配置")

    material = db.get_material(material_id)
    if not material:
        raise HTTPException(status_code=404, detail="所选素材不存在")
    if material.get("type") != "image":
        raise HTTPException(status_code=400, detail="Tripo3D 生成需要选择图片素材")

    image_path = DIRS["upload"] / material["filename"]
    if not image_path.exists():
        raise HTTPException(status_code=404, detail="素材图片文件不存在")

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=15)) as client:
        uploaded = await _upload_image_to_tripo(client, image_path)
        file_type = image_path.suffix.lower().lstrip(".") or "jpg"
        file_payload = {"type": file_type, "file_token": uploaded["token"]}

        multiview_payload = {
            "type": "generate_multiview_image",
            "file": file_payload,
            "image": file_payload,
            "model_version": TRIPO_MODEL_VERSION,
            "enable_image_autofix": TRIPO_ENABLE_AUTOFIX,
        }
        task_id = ""
        task_data = None
        try:
            print("[Tripo3D] submitting multiview image task")
            multiview_task_id = await _create_tripo_task(client, multiview_payload)
            multiview_data = await _poll_tripo_task(client, multiview_task_id)
            multiview_token = _find_first_key(multiview_data, ("image_token", "file_token", "token"))
            multiview_url = _find_first_key(multiview_data, ("image_url", "url"))
            model_payload = {
                "type": "multiview_to_model",
                "model_version": TRIPO_MODEL_VERSION,
                "geometry_quality": TRIPO_GEOMETRY_QUALITY,
                "texture": TRIPO_ENABLE_TEXTURE,
                "pbr": TRIPO_ENABLE_TEXTURE,
                "auto_size": TRIPO_AUTO_SIZE,
            }
            if multiview_token:
                model_payload["file"] = {"type": "png", "file_token": multiview_token}
            elif multiview_url:
                model_payload["image_url"] = multiview_url
            else:
                model_payload["file"] = file_payload
            print("[Tripo3D] submitting multiview-to-model task")
            task_id = await _create_tripo_task(client, model_payload)
            task_data = await _poll_tripo_task(client, task_id)
        except HTTPException as exc:
            print(f"[Tripo3D] multiview flow unavailable, fallback to image_to_model: {exc.detail}")
            image_payload = {
                "type": "image_to_model",
                "file": file_payload,
                "model_version": TRIPO_MODEL_VERSION,
                "geometry_quality": TRIPO_GEOMETRY_QUALITY,
                "texture": TRIPO_ENABLE_TEXTURE,
                "pbr": TRIPO_ENABLE_TEXTURE,
                "auto_size": TRIPO_AUTO_SIZE,
                "enable_image_autofix": TRIPO_ENABLE_AUTOFIX,
            }
            task_id = await _create_tripo_task(client, image_payload)
            task_data = await _poll_tripo_task(client, task_id)

        downloaded = await _download_model_urls(client, task_id, task_data, output_dir)
        if not downloaded:
            raise HTTPException(status_code=502, detail=f"Tripo3D 任务完成但没有找到可下载模型文件: {task_data}")
        primary = downloaded.get("stl") or downloaded.get("3mf") or downloaded.get("glb") or next(iter(downloaded.values()))
        return {
            "task_id": task_id,
            "primary_filename": primary,
            "format": os.path.splitext(primary)[1].lstrip(".") or "model",
            "files": downloaded,
        }


def _latest_story_text() -> str:
    stories = db.get_stories()
    if stories:
        return stories[0].get("content", "")
    return db.get_ip_profile().get("bio", "")


def _pick_default_ip_image() -> Optional[Path]:
    if DEFAULT_IP_IMAGE_DIR.exists():
        candidates = sorted(
            p for p in DEFAULT_IP_IMAGE_DIR.iterdir()
            if p.is_file() and p.suffix.lower() in DEFAULT_IP_IMAGE_EXTENSIONS
        )
        if candidates:
            return candidates[0]

    profile = db.get_ip_profile()
    avatar_url = profile.get("avatar_url") or ""
    if avatar_url.startswith("/static/upload/"):
        avatar_path = DIRS["upload"] / avatar_url.rsplit("/", 1)[-1]
        if avatar_path.exists():
            return avatar_path
    return None


def _file_to_public_media_url(path: Path) -> str:
    """HappyHorse downloads media from a public URL; local paths and data URIs are not accepted."""
    media_base_url = os.getenv("HAPPYHORSE_MEDIA_BASE_URL", "").strip().rstrip("/")
    if not media_base_url:
        raise HTTPException(
            status_code=400,
            detail=(
                "HappyHorse 需要可公网下载的首帧图片 URL。请把 Milky 的 IP 形象上传到 OSS/图床，"
                "然后设置环境变量 HAPPYHORSE_MEDIA_BASE_URL 为该公开目录地址；本地文件和 localhost 无法被阿里服务端下载。"
            ),
        )
    public_dir = BASE_DIR / "static" / "happyhorse-media"
    public_dir.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower() if path.suffix.lower() in DEFAULT_IP_IMAGE_EXTENSIONS else ".png"
    public_name = f"first_frame{ext}"
    public_path = public_dir / public_name
    if not public_path.exists() or public_path.stat().st_mtime < path.stat().st_mtime:
        shutil.copy2(path, public_path)
    return f"{media_base_url}/static/happyhorse-media/{public_name}"


def _build_happyhorse_video_prompt(user_prompt: str, source_image: Optional[Path]) -> str:
    fallback_story = _latest_story_text()
    base = user_prompt.strip() or (
        "Create a cute Japanese anime cartoon highlight video of Milky, a charming pet IP character. "
        "Keep the character consistent with the reference image, warm, playful, polished, cinematic, "
        "with gentle camera motion, soft lighting, expressive eyes, and a cozy storybook feeling."
    )
    if fallback_story:
        return f"{base}\nCharacter bio and story inspiration: {fallback_story[:800]}"
    return base


def _dashscope_error_detail(provider: str, status_code: int, payload: dict) -> str:
    error_text = str(payload)
    lower = error_text.lower()
    if any(word in lower for word in ("quota", "credit", "insufficient", "余额", "额度")):
        return f"{provider} 账号额度不足，无法生成视频。请检查 MaaS/DashScope 额度或更换有额度的 API key。原始返回: {payload}"
    return f"{provider} API 请求失败({status_code}): {payload}"


async def _dashscope_request(client: httpx.AsyncClient, method: str, path: str, **kwargs) -> dict:
    headers = kwargs.pop("headers", {})
    headers.update({"Authorization": f"Bearer {HAPPYHORSE_API_KEY}"})
    url = f"{HAPPYHORSE_BASE_URL}{path}"
    try:
        resp = await client.request(method, url, headers=headers, **kwargs)
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout, httpx.NetworkError) as exc:
        raise HTTPException(
            status_code=504,
            detail=f"无法连接 HappyHorse/DashScope API（{HAPPYHORSE_BASE_URL}）。请检查网络、DNS 或代理后重试。原始错误: {exc}"
        )
    try:
        payload = resp.json()
    except Exception:
        payload = {"raw": resp.text}
    if resp.status_code >= 400:
        raise HTTPException(status_code=502, detail=_dashscope_error_detail("HappyHorse", resp.status_code, payload))
    if isinstance(payload, dict) and payload.get("code") not in (None, 0, 200, "0", "200"):
        raise HTTPException(status_code=502, detail=f"HappyHorse API 返回错误: {payload}")
    return payload


def _extract_task_id(payload: dict) -> Optional[str]:
    output = payload.get("output") if isinstance(payload, dict) else None
    if isinstance(output, dict):
        value = output.get("task_id") or output.get("taskId") or output.get("id")
        if value:
            return str(value)
    value = payload.get("task_id") or payload.get("taskId") or payload.get("id")
    return str(value) if value else None


def _extract_task_status(payload: dict) -> str:
    output = payload.get("output") if isinstance(payload, dict) else None
    if isinstance(output, dict):
        value = output.get("task_status") or output.get("status") or output.get("state")
        if value:
            return str(value).upper()
    return str(payload.get("task_status") or payload.get("status") or payload.get("state") or "").upper()


def _first_video_output_url(task: dict) -> Optional[str]:
    preferred_keys = {"video_url", "video", "url", "uri"}
    queue: list[Any] = [task]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for key in preferred_keys:
                value = current.get(key)
                if isinstance(value, str) and value.startswith(("http://", "https://")):
                    return value
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
        elif isinstance(current, str) and current.startswith(("http://", "https://")):
            lowered = current.split("?", 1)[0].lower()
            if lowered.endswith((".mp4", ".mov", ".webm")):
                return current
    urls = _collect_urls(task)
    for url in urls:
        if url.split("?", 1)[0].lower().endswith((".mp4", ".mov", ".webm")):
            return url
    return urls[0] if urls else None


async def _poll_happyhorse_task(client: httpx.AsyncClient, task_id: str) -> dict:
    deadline = asyncio.get_running_loop().time() + HAPPYHORSE_TASK_TIMEOUT_SECONDS
    last_payload = None
    while asyncio.get_running_loop().time() < deadline:
        payload = await _dashscope_request(client, "GET", f"/tasks/{task_id}")
        last_payload = payload
        status = _extract_task_status(payload)
        print(f"[HappyHorse] task={task_id} status={status or 'UNKNOWN'}")
        if status in {"SUCCEEDED", "SUCCESS", "COMPLETED", "COMPLETE", "FINISHED"}:
            return payload
        if status in {"FAILED", "FAILURE", "CANCELLED", "CANCELED", "ERROR"}:
            raise HTTPException(status_code=502, detail=f"HappyHorse 视频生成失败: {payload}")
        await asyncio.sleep(HAPPYHORSE_POLL_INTERVAL_SECONDS)
    raise HTTPException(status_code=504, detail=f"HappyHorse 视频生成超时: {last_payload}")


def _video_size_from_ratio(ratio: str) -> str:
    value = (ratio or HAPPYHORSE_VIDEO_RATIO).strip()
    if "*" in value:
        return value
    if ":" in value:
        return value.replace(":", "*")
    return HAPPYHORSE_VIDEO_SIZE


async def call_happyhorse_video_api(prompt: str, duration: int, ratio: str, output_dir: Path) -> dict:
    if not HAPPYHORSE_API_KEY:
        raise HTTPException(status_code=500, detail="HappyHorse API Key 未配置")

    source_image = _pick_default_ip_image()
    if not source_image:
        raise HTTPException(status_code=404, detail=f"未找到 IP 形象图片，请检查目录: {DEFAULT_IP_IMAGE_DIR}")

    final_prompt = _build_happyhorse_video_prompt(prompt, source_image)
    payload = {
        "model": HAPPYHORSE_MODEL,
        "input": {
            "prompt": final_prompt,
            "media": [
                {
                    "type": "first_frame",
                    "url": _file_to_public_media_url(source_image),
                }
            ],
        },
        "parameters": {
            "size": _video_size_from_ratio(ratio),
            "duration": duration or HAPPYHORSE_VIDEO_DURATION,
        },
    }

    async with httpx.AsyncClient(timeout=httpx.Timeout(120, connect=20)) as client:
        task = await _dashscope_request(
            client,
            "POST",
            "/services/aigc/video-generation/video-synthesis",
            json=payload,
            headers={"Content-Type": "application/json", "X-DashScope-Async": "enable"},
        )
        task_id = _extract_task_id(task)
        if not task_id:
            raise HTTPException(status_code=502, detail=f"HappyHorse 未返回任务 ID: {task}")
        result = await _poll_happyhorse_task(client, task_id)
        video_url = _first_video_output_url(result)
        if not video_url:
            raise HTTPException(status_code=502, detail=f"HappyHorse 任务完成但未返回视频 URL: {result}")
        resp = await client.get(video_url, follow_redirects=True)
        if resp.status_code >= 400:
            resp = await client.get(video_url, headers={"Authorization": f"Bearer {HAPPYHORSE_API_KEY}"}, follow_redirects=True)
        resp.raise_for_status()

    output_dir.mkdir(parents=True, exist_ok=True)
    ext = os.path.splitext(video_url.split("?", 1)[0])[1].lower() or ".mp4"
    if ext not in {".mp4", ".mov", ".webm"}:
        ext = ".mp4"
    filename = f"happyhorse_{task_id}{ext}"
    target = output_dir / filename
    async with aiofiles.open(target, "wb") as f:
        await f.write(resp.content)
    return {"filename": filename, "task_id": task_id, "prompt": final_prompt, "source_image": str(source_image), "provider": "HappyHorse"}


# ============================================================
# API 路由
# ============================================================

# ---- 健康检查 ----
@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now().isoformat(), "service": "宠IP工坊 API"}


# ==================== 素材管理 ====================

@app.post("/api/upload")
async def upload_material(
    file: UploadFile = File(...),
    behavior: str = Form("未知"),
    emotion: str = Form("未知"),
):
    """上传宠物素材（图片/视频），可选指定行为与情绪标签"""
    # 校验文件类型
    allowed_types = ("image/jpeg", "image/png", "image/gif", "image/webp", "video/mp4", "video/quicktime")
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的文件类型: {file.content_type}")

    # 生成唯一文件名并保存
    filename = generate_unique_filename(file.filename or "upload.png")
    filepath = DIRS["upload"] / filename

    content = await file.read()
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(content)

    # 入库
    item = db.add_material(
        filename=filename,
        original_name=file.filename or filename,
        file_size=len(content),
        file_type=file.content_type or "image/png",
        behavior=behavior,
        emotion=emotion,
    )

    return {"code": 200, "message": "上传成功", "data": item}


@app.get("/api/materials")
async def list_materials(
    search: str = "",
    behavior: str = "",
    emotion: str = "",
    scene: str = "",
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    """获取素材列表，支持搜索和筛选"""
    materials = db.get_materials()

    # 搜索
    if search:
        search_lower = search.lower()
        materials = [
            m for m in materials
            if search_lower in m["behavior"].lower()
            or search_lower in m["emotion"].lower()
            or search_lower in m["original_name"].lower()
            or any(search_lower in t.lower() for t in m["tags"])
        ]

    # 行为筛选
    if behavior:
        materials = [m for m in materials if m["behavior"] == behavior]

    # 情绪筛选
    if emotion:
        materials = [m for m in materials if m["emotion"] == emotion]

    # 来源筛选
    if scene:
        materials = [m for m in materials if m.get("scene") == scene]

    total = len(materials)
    start = (page - 1) * page_size
    end = start + page_size

    return {
        "code": 200,
        "data": materials[start:end],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


@app.get("/api/materials/{material_id}")
async def get_material(material_id: int):
    """获取单个素材详情"""
    item = db.get_material(material_id)
    if not item:
        raise HTTPException(status_code=404, detail="素材不存在")
    return {"code": 200, "data": item}


@app.delete("/api/materials/{material_id}")
async def delete_material(material_id: int):
    """删除素材"""
    item = db.delete_material(material_id)
    if not item:
        raise HTTPException(status_code=404, detail="素材不存在")
    return {"code": 200, "message": "删除成功"}


@app.post("/api/materials/{material_id}/favorite")
async def toggle_favorite(material_id: int):
    """切换素材收藏状态"""
    item = db.toggle_favorite(material_id)
    if not item:
        raise HTTPException(status_code=404, detail="素材不存在")
    return {"code": 200, "message": "操作成功", "data": item}


@app.post("/api/materials/{material_id}/tags")
async def add_tags(material_id: int, tags: list[str] = Form(...)):
    """为素材添加标签"""
    item = db.update_material_tags(material_id, tags)
    if not item:
        raise HTTPException(status_code=404, detail="素材不存在")
    return {"code": 200, "message": "标签已更新", "data": item}


class UpdateMaterialRequest(BaseModel):
    behavior: Optional[str] = None
    emotion: Optional[str] = None
    tags: Optional[list[str]] = None

@app.put("/api/materials/{material_id}")
async def update_material(material_id: int, req: UpdateMaterialRequest):
    """更新素材的行为/情绪/标签等元数据"""
    item = db.get_material(material_id)
    if not item:
        raise HTTPException(status_code=404, detail="素材不存在")
    if req.behavior is not None:
        item["behavior"] = req.behavior
    if req.emotion is not None:
        item["emotion"] = req.emotion
    if req.tags is not None:
        existing = set(item["tags"])
        for t in req.tags:
            existing.add(t)
        item["tags"] = list(existing)
    db._save()
    return {"code": 200, "message": "更新成功", "data": item}


@app.get("/api/download/material/{filename}")
async def download_material(filename: str):
    """下载素材文件"""
    filepath = DIRS["upload"] / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(filepath), filename=filename)


# ==================== AI 生成：IP 形象 ====================

@app.post("/api/generate/ip")
async def generate_ip(req: GenerateIPRequest):
    """
    调用即梦 AI API 生成宠物 IP 形象。
    生成多张不同角度的 IP 设计图并保存到 gen_ip 目录。
    """
    output_dir = DIRS["gen_ip"]
    filenames = await call_jimeng_api(req.prompt, req.style, output_dir, "ip")

    items = []
    for fname in filenames:
        item = db.add_ip_image(fname, req.prompt, req.style)
        items.append(item)

    return {"code": 200, "message": "IP 形象生成成功", "data": items}


@app.get("/api/generate/ip")
async def list_ip_images():
    """获取已生成的 IP 形象列表"""
    return {"code": 200, "data": db.get_ip_images()}


@app.get("/api/download/ip/{filename}")
async def download_ip_image(filename: str):
    """下载 IP 形象图片"""
    filepath = DIRS["gen_ip"] / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(filepath), filename=filename)


# ==================== IP 档案：简介与照片 ====================

@app.get("/api/ip-profile")
async def get_ip_profile():
    """获取当前 IP 工坊角色档案"""
    return {"code": 200, "data": db.get_ip_profile()}


@app.put("/api/ip-profile")
async def update_ip_profile(req: IPProfileUpdate):
    """保存 IP 工坊角色名称、风格、简介和头像信息"""
    if hasattr(req, "model_dump"):
        data = req.model_dump(exclude_unset=True)
    else:
        data = req.dict(exclude_unset=True)
    profile = db.update_ip_profile(data)
    return {"code": 200, "message": "IP 档案已保存", "data": profile}


@app.post("/api/ip-profile/photo")
async def upload_ip_profile_photo(file: UploadFile = File(...)):
    """上传 IP 工坊照片，并设置为当前角色头像"""
    allowed_types = ("image/jpeg", "image/png", "image/gif", "image/webp")
    if file.content_type and file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail=f"不支持的图片类型: {file.content_type}")

    filename = generate_unique_filename(file.filename or "ip-profile.png")
    filepath = DIRS["upload"] / filename

    content = await file.read()
    async with aiofiles.open(filepath, "wb") as f:
        await f.write(content)

    item = db.add_material(
        filename=filename,
        original_name=file.filename or filename,
        file_size=len(content),
        file_type=file.content_type or "image/png",
        behavior="IP头像",
        emotion="未知",
    )
    profile = db.update_ip_profile({
        "avatar_url": item["url"],
        "avatar_material_id": item["id"],
    })

    return {
        "code": 200,
        "message": "照片上传成功",
        "data": {
            "profile": profile,
            "material": item,
        },
    }

# ==================== AI 生成：漫画 ====================
@app.post("/api/generate/comic")
async def generate_comic(req: GenerateComicRequest):
    """
    调用硅基流动「Qwen3-VL 看图 + Tongyi-MAI/Z-Image 图生图」生成图片（复刻自 main(1).py）。
    真实 API 为主路径；仅当素材库无可用图片时，才回退到预设占位图。
    """
    print("\n【COMIC API】漫画接口已触发")
    output_dir = DIRS["gen_comic"]

    # 预设漫画图片路径（仅作为无素材时的兜底，不再拦截真实 API）
    preset_file = "漫画.png"
    preset_path = output_dir / preset_file

    latest_image = _latest_material_image_path()
    if latest_image:
        # 风格参考图（static/风格参考图.png），若不存在则忽略
        style_ref_path = BASE_DIR / "static" / "风格参考图.png"
        if not style_ref_path.exists():
            style_ref_path = None

        # Step 1: Qwen3-VL 看风格参考图 + 素材图，生成综合提示词
        print("【Step 1】Qwen3-VL 分析风格+素材图...")
        story_text = await call_siliconflow_vl_api(
            prompt=req.prompt,
            image_path=latest_image,
            style=req.style,
            style_ref_path=style_ref_path,
        )
        # Step 2: 图生图（只传素材图，VL 输出文字中已含风格描述）
        print("【Step 2】Tongyi-MAI/Z-Image 图生图...")
        with open(latest_image, "rb") as f:
            ref_b64 = base64.b64encode(f.read()).decode()
        ext = latest_image.suffix.lower()
        mime = "image/png" if ext == ".png" else "image/jpeg"
        ref_data_uri = f"data:{mime};base64,{ref_b64}"
        filenames = await call_siliconflow_api(
            prompt=story_text,
            style=req.style,
            ref_image=ref_data_uri,
            output_dir=output_dir,
            prefix="comic",
        )
        item = db.add_comic(filenames, req.prompt)
        return {"code": 200, "message": "漫画生成成功", "data": item}

    # 无可用素材图：优先用预设占位图，否则触发 SVG 占位兜底
    if preset_path.exists():
        await asyncio.sleep(3)
        filenames = [preset_file]
    else:
        filenames = await call_siliconflow_api(
            prompt=req.prompt,
            style=req.style,
            ref_image="",
            output_dir=output_dir,
            prefix="comic",
        )
    item = db.add_comic(filenames, req.prompt)
    return {"code": 200, "message": "漫画生成成功", "data": item}


@app.get("/api/download/comic/{filename}")
async def download_comic(filename: str):
    """下载漫画图片"""
    filepath = DIRS["gen_comic"] / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(filepath), filename=filename)


# ==================== AI 生成：IP 周边设计 ====================

GOODS_NAMES = {
    "blindbox": "盲盒", "keychain": "钥匙扣", "fridge": "冰箱贴", "mug": "马克杯", "bag": "帆布包",
}

# 每个产品的「样例图」——用来告诉 FLUX 产品长什么样、什么风格
GOODS_SAMPLES = {
    "mug": ("IP 周边", "马克杯1.png"),
    "fridge": ("IP 周边", "定制冰箱贴.png"),
    "bag": ("IP 周边", "定制帆布包.png"),
    "blindbox": ("盲盒", "盲盒1.png"),
    "keychain": ("盲盒", "盲盒1.png"),
}


def _goods_sample_path(product: str) -> Optional[Path]:
    entry = GOODS_SAMPLES.get(product)
    if not entry: return None
    subdir, fname = entry
    p = BASE_DIR / "Milky酱" / subdir / fname
    return p if p.exists() else None


class GenerateGoodsRequest(BaseModel):
    product: str = "mug"
    prompt: str = ""


@app.post("/api/generate/goods")
async def generate_goods(req: GenerateGoodsRequest):
    print("\n【GOODS】Seedream 周边设计")
    output_dir = DIRS["gen_ip"]
    product_cn = GOODS_NAMES.get(req.product, "周边")

    latest_image = _latest_material_image_path()
    if not latest_image:
        return {"code": 400, "message": "素材库没有图片"}

    # 准备两张图
    sample_path = _goods_sample_path(req.product)
    images = []
    if sample_path:
        with open(sample_path, "rb") as f:
            images.append(f"data:image/png;base64,{base64.b64encode(f.read()).decode()}")
    with open(latest_image, "rb") as f:
        mime = "image/png" if latest_image.suffix.lower() == ".png" else "image/jpeg"
        images.append(f"data:{mime};base64,{base64.b64encode(f.read()).decode()}")

    prompt = req.prompt or (
        f"图1是产品参考图，图2是猫的照片。"
        f"请学习图1的产品风格、材质、光影和构图，保持这些完全不变。"
        f"把图1产品上原本的猫的图案，替换成图2的猫——毛色、花纹、眼睛、姿势都要和图2一模一样。"
        f"将图2的猫画成软萌可爱的卡通IP风格，线条柔和，配色温暖，"
        f"风格与图1原本的产品图保持一致。纯白背景，高清产品摄影。"
    )

    print("  Seedream 生图中...")
    try:
        headers = {
            "Authorization": f"Bearer {ARK_API_KEY}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": ARK_MODEL,
            "prompt": prompt,
            "size": "1920x1920",
            "response_format": "url",
            "image": images,  # 两张图：产品参考图 + 猫照片
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{ARK_BASE_URL}/images/generations",
                json=payload,
                headers=headers
            )
            if resp.status_code != 200:
                print(f"  ❌ Seedream 返回 {resp.status_code}: {resp.text[:500]}")
                resp.raise_for_status()
            data = resp.json()
        image_url = data["data"][0]["url"]
        async with httpx.AsyncClient() as hc:
            ir = await hc.get(image_url); ir.raise_for_status()
        fname = f"goods_{uuid.uuid4().hex[:8]}.png"
        (output_dir / fname).write_bytes(ir.content)
        print(f"  \u2705 {fname}")
    except Exception as e:
        print(f"  \u274c {e}")
        fname = f"goods_{uuid.uuid4().hex[:8]}.svg"
        generate_placeholder_image(output_dir / fname, 512, 512, product_cn)

    db.add_ip_image(fname, prompt, f"周边\u00b7{product_cn}") if fname else None
    return {"code": 200, "message": "生成成功", "data": {"product": req.product, "name": product_cn, "filename": fname, "url": f"/static/gen_ip/{fname}" if fname else ""}}


@app.get("/api/download/goods/{filename}")
async def download_goods(filename: str):
    """下载周边设计图片"""
    filepath = DIRS["gen_ip"] / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(filepath), filename=filename)


@app.get("/api/generate/goods")
async def list_goods():
    """获取周边设计历史列表"""
    all_items = db.get_ip_images()
    goods = [i for i in all_items if "周边" in i.get("style", "")]
    return {"code": 200, "data": goods}


@app.delete("/api/generate/goods/{goods_id}")
async def delete_goods(goods_id: int):
    """删除指定周边设计"""
    items = db.get_ip_images()
    for i, item in enumerate(items):
        if item["id"] == goods_id and "周边" in item.get("style", ""):
            filepath = DIRS["gen_ip"] / item["filename"]
            if filepath.exists():
                filepath.unlink()
            items.pop(i)
            db._save()
            return {"code": 200, "message": "删除成功"}
    raise HTTPException(status_code=404, detail="周边不存在")


@app.get("/api/generate/story")
async def list_stories():
    """获取已生成的故事列表"""
    return {"code": 200, "data": db.get_stories()}


# ==================== AI 生成：精华短片（HappyHorse） ====================

@app.post("/api/generate/video")
async def generate_video(req: GenerateVideoRequest):
    """异步提交精华短片生成任务，立即返回 task_id。"""
    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "pending", "result": None, "error": None}
    asyncio.create_task(_run_video_task(task_id, req))
    return {"code": 200, "task_id": task_id, "message": "视频生成任务已提交，请轮询状态"}


async def _run_video_task(task_id: str, req: GenerateVideoRequest):
    """后台执行 HappyHorse 视频生成"""
    try:
        _tasks[task_id]["status"] = "processing"
        result = await call_happyhorse_video_api(req.prompt, req.duration, req.ratio, DIRS["gen_video"])
        item = db.add_video(
            filename=result["filename"],
            prompt=result["prompt"],
            source_image=result["source_image"],
            provider_task_id=result["task_id"],
            provider=result.get("provider", "HappyHorse"),
        )
        _tasks[task_id] = {"status": "completed", "result": item, "error": None}
    except Exception as e:
        _tasks[task_id] = {"status": "failed", "result": None, "error": str(e)}
        print(f"[_run_video_task] {task_id} 失败: {e}")


@app.get("/api/generate/video/task/{task_id}")
async def get_video_task_status(task_id: str):
    """查询视频生成异步任务状态"""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"code": 200, "data": task}


@app.get("/api/generate/video")
async def list_videos():
    """获取已生成的精华短片列表"""
    return {"code": 200, "data": db.get_videos()}


# ==================== AI 生成：3D 模型 ====================

@app.post("/api/generate/3d")
async def generate_3d(req: Generate3DRequest):
    """异步提交 3D 生成任务，立即返回 task_id。"""
    target_material_id = req.material_ids[0] if req.material_ids else import_default_3d_source_material()
    if not target_material_id:
        materials = [m for m in db.get_materials() if m.get("type") == "image"]
        target_material_id = materials[0]["id"] if materials else None
    if not target_material_id:
        raise HTTPException(status_code=400, detail="请选择至少一张宠物图片素材")

    task_id = str(uuid.uuid4())
    _tasks[task_id] = {"status": "pending", "result": None, "error": None}
    asyncio.create_task(_run_tripo_task(task_id, int(target_material_id), DIRS["gen_3d"], req))
    return {"code": 200, "task_id": task_id, "message": "3D 生成任务已提交，请轮询状态"}


async def _run_tripo_task(task_id: str, material_id: int, output_dir: Path, req: Generate3DRequest):
    """后台执行 Tripo3D 生成，完成后更新任务状态"""
    try:
        _tasks[task_id]["status"] = "processing"
        result = await call_tripo_image_to_3d(material_id, output_dir)
        item = db.add_3d_model(
            filename=result["primary_filename"],
            prompt=req.prompt,
            pose=req.pose,
            format_type=result["format"],
            extra={
                "stl_filename": result["files"].get("stl", ""),
                "3mf_filename": result["files"].get("3mf", ""),
                "glb_filename": result["files"].get("glb", ""),
                "tripo_task_id": result["task_id"],
                "source_material_id": material_id,
            },
        )
        _tasks[task_id] = {"status": "completed", "result": item, "error": None}
    except Exception as e:
        _tasks[task_id] = {"status": "failed", "result": None, "error": str(e)}
        print(f"[_run_tripo_task] {task_id} 失败: {e}")


@app.get("/api/generate/3d/task/{task_id}")
async def get_tripo_task_status(task_id: str):
    """查询 3D 生成异步任务状态"""
    task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    return {"code": 200, "data": task}


@app.get("/api/generate/3d")
async def list_3d_models():
    """获取已生成的 3D 模型列表"""
    return {"code": 200, "data": db.get_3d_models()}


@app.get("/api/download/3d/{filename}")
async def download_3d_model(filename: str):
    """下载 3D 模型文件（STL/3MF）"""
    filepath = DIRS["gen_3d"] / filename
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(str(filepath), filename=filename, media_type="application/octet-stream")


# ==================== 3D 打印文件导出 ====================

class Export3DRequest(BaseModel):
    model_config = {"protected_namespaces": ()}
    model_id: int
    format: str = "stl"  # stl 或 3mf
    size: str = "10cm"
    material: str = "PLA"
    precision: str = "0.2mm"


@app.post("/api/export/3d")
async def export_3d_file(req: Export3DRequest):
    """
    导出 3D 打印文件。
    如果 format=stl，直接返回已有文件；format=3mf 则需要转换。
    """
    # 查找模型
    models = db.get_3d_models()
    model = None
    for m in models:
        if m["id"] == req.model_id:
            model = m
            break

    if not model:
        raise HTTPException(status_code=404, detail="模型不存在")

    requested_format = req.format.lower()
    if requested_format == "glb" and model.get("glb_filename"):
        export_filename = model["glb_filename"]
    elif requested_format == "3mf" and model.get("3mf_filename"):
        export_filename = model["3mf_filename"]
    elif requested_format == "stl" and model.get("stl_filename"):
        export_filename = model["stl_filename"]
    else:
        export_filename = model.get("glb_filename") or model.get("filename", "")

    export_path = DIRS["gen_3d"] / export_filename
    if not export_filename or not export_path.exists():
        raise HTTPException(status_code=404, detail="模型文件不存在，请重新生成")

    actual_format = os.path.splitext(export_filename)[1].lstrip(".") or model.get("format", requested_format)
    return FileResponse(
        str(export_path),
        filename=export_filename,
        media_type="application/octet-stream",
        headers={
            "X-Export-Format": req.format,
            "X-Export-Actual-Format": actual_format,
            "X-Export-Filename": export_filename,
            "X-Export-Size": req.size,
            "X-Export-Material": req.material,
            "X-Export-Precision": req.precision,
        }
    )


# ==================== 首页 · 本周人气周边 ====================
# image 为预留的真实图片槽位：把真实图放到 static/goods/{id}.png 即可生效；
# 缺图时前端自动回退到 emoji 占位。
# 顺序：盲盒为主，其后马克杯等其他周边；图片不重复
FEATURED_GOODS = [
    {"id": 1, "name": "盲盒1",       "price": "¥59",  "badge": "新品", "emoji": "📦", "image": "/static/blindbox/盲盒1.png"},
    {"id": 2, "name": "盲盒2",       "price": "¥59",  "badge": "",     "emoji": "📦", "image": "/static/blindbox/盲盒2.png"},
    {"id": 3, "name": "盲盒3",       "price": "¥59",  "badge": "热销", "emoji": "📦", "image": "/static/blindbox/盲盒3.png"},
    {"id": 4, "name": "盲盒4",       "price": "¥59",  "badge": "",     "emoji": "📦", "image": "/static/blindbox/盲盒4.png"},
    {"id": 5, "name": "限定盲盒",     "price": "¥99",  "badge": "限量", "emoji": "📦", "image": "/static/blindbox/限定盲盒.png"},
    {"id": 6, "name": "马克杯1",      "price": "¥49",  "badge": "",     "emoji": "☕", "image": "/static/ipmerch/马克杯1.png"},
    {"id": 7, "name": "马克杯2",      "price": "¥49",  "badge": "",     "emoji": "☕", "image": "/static/ipmerch/马克杯2.png"},
    {"id": 8, "name": "定制帆布包",   "price": "¥69",  "badge": "",     "emoji": "🎒", "image": "/static/ipmerch/定制帆布包.png"},
]


@app.get("/api/featured-goods")
async def list_featured_goods():
    """首页『本周人气周边』数据；image 字段为真实图片槽位（缺图前端回退 emoji）。"""
    goods = []
    for g in FEATURED_GOODS:
        item = dict(g)
        # image 形如 /static/blindbox/盲盒1.png → 映射到磁盘路径判断是否存在
        rel = g.get("image", "").lstrip("/").replace("static/", "", 1)
        img_path = BASE_DIR / "static" / rel
        item["has_image"] = img_path.exists()
        goods.append(item)
    return {"code": 200, "data": goods}


# ==================== 设备管理 ====================

@app.get("/api/devices")
async def list_devices():
    """获取设备列表"""
    return {"code": 200, "data": db.get_devices()}


@app.post("/api/devices")
async def add_device(device: DeviceBase):
    """添加新设备"""
    item = db.add_device(device.name, device.group)
    return {"code": 200, "message": "设备添加成功", "data": item}


@app.put("/api/devices/{device_id}")
async def update_device(device_id: int, data: dict):
    """更新设备信息"""
    item = db.update_device(device_id, data)
    if not item:
        raise HTTPException(status_code=404, detail="设备不存在")
    return {"code": 200, "data": item}


@app.post("/api/devices/{device_id}/sync")
async def sync_device(device_id: int):
    """同步设备数据"""
    item = db.update_device(device_id, {"last_sync": "刚刚", "status": "online"})
    if not item:
        raise HTTPException(status_code=404, detail="设备不存在")
    # 模拟同步延迟
    await asyncio.sleep(0.5)
    return {"code": 200, "message": f"设备 {item['name']} 同步成功", "data": item}


# ==================== 统计概览 ====================

@app.get("/api/stats")
async def get_stats():
    """获取仪表盘统计数据"""
    materials = db.get_materials()
    devices = db.get_devices()
    return {
        "code": 200,
        "data": {
            "total_devices": len(devices),
            "online_devices": sum(1 for d in devices if d["status"] == "online"),
            "total_materials": len(materials),
            "total_images": sum(1 for m in materials if m["type"] == "image"),
            "total_videos": sum(1 for m in materials if m["type"] == "video"),
            "total_favorites": sum(1 for m in materials if m["favorite"]),
            "today_captures": 1248,
            "record_hours": 86,
            "device_status": "良好",
        }
    }


# ============================================================
# 启动时导入 Milky 相册种子素材（仅首次）
# ============================================================
seed_demo_materials()


# ============================================================
# 启动入口
# ============================================================
if __name__ == "__main__":
    import uvicorn
    print(f"🚀 宠 IP 工坊 API 启动中...")
    print(f"   📂 素材上传目录: {DIRS['upload']}")
    print(f"   🎨 IP 形象目录: {DIRS['gen_ip']}")
    print(f"   📚 漫画生成目录: {DIRS['gen_comic']}")
    print(f"   🖨️ 3D 模型目录: {DIRS['gen_3d']}")
    print(f"   🌐 前端页面: http://localhost:8000/")
    print(f"   📖 API 文档: http://localhost:8000/docs")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

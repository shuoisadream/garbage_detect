import gradio as gr
import mindspore as ms
from mindspore import Tensor
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
import math
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
from collections import Counter
import json

matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'WenQuanYi Micro Hei']
matplotlib.rcParams['axes.unicode_minus'] = False

CONFIG_PATH = "./mindyolo/configs/yolov11/yolov11n_garbage.yaml"
WEIGHT_PATH = "./best.ckpt"
CONF_THRES  = 0.25
IOU_THRES   = 0.65
IMG_SIZE    = 640
CONF_FREE   = True

OUTPUT_DIR = "./outputs"
HISTORY_FILE = "./outputs/detection_history.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['BIODEGRADABLE', 'CARDBOARD', 'GLASS', 'METAL', 'PAPER', 'PLASTIC']
CLASS_NAMES_CN = ['生物降解', '纸板', '玻璃', '金属', '纸张', '塑料']

COLORS = [
    (16,  185, 129),
    (59,  130, 246),
    (34,  197, 94),
    (6,   182, 212),
    (99,  102, 241),
    (14,  165, 233),
]

GARBAGE_GUIDE = {
    'BIODEGRADABLE': {
        'cn_name': '生物降解垃圾',
        'color': '#10b981',
        'icon': '🌿',
        'description': '可自然分解的有机废弃物',
        'examples': ['剩菜剩饭', '果皮果核', '茶叶渣', '花卉植物', '蛋壳'],
        'disposal': ['投放到厨余垃圾桶（绿色）', '可用于堆肥处理', '避免混入塑料袋等杂质', '沥干水分后投放更佳'],
        'tips': '厨余垃圾经处理后可转化为有机肥料，实现资源循环利用。'
    },
    'CARDBOARD': {
        'cn_name': '纸板垃圾',
        'color': '#3b82f6',
        'icon': '📦',
        'description': '可回收的纸质包装材料',
        'examples': ['快递纸箱', '鞋盒', '牙膏盒', '纸板包装'],
        'disposal': ['投放到可回收物桶（蓝色）', '拆解压扁后投放节省空间', '保持干燥，避免污染', '去除胶带和标签更佳'],
        'tips': '回收一吨废纸可生产约850公斤再生纸，节约木材3立方米。'
    },
    'GLASS': {
        'cn_name': '玻璃垃圾',
        'color': '#22c55e',
        'icon': '🫙',
        'description': '各类玻璃制品废弃物',
        'examples': ['玻璃瓶', '玻璃罐', '窗户玻璃', '玻璃杯'],
        'disposal': ['投放到可回收物桶（蓝色）', '小心轻放，避免破碎', '瓶内液体需倒干净', '区分有色玻璃和无色玻璃'],
        'tips': '玻璃可无限次回收利用，回收后熔制温度低，能源消耗少。'
    },
    'METAL': {
        'cn_name': '金属垃圾',
        'color': '#06b6d4',
        'icon': '🥫',
        'description': '各类金属制品废弃物',
        'examples': ['易拉罐', '罐头盒', '金属厨具', '铝制品'],
        'disposal': ['投放到可回收物桶（蓝色）', '清洗后再投放', '压缩易拉罐节省空间', '区分铁制和铝制金属'],
        'tips': '回收一吨废铝可节约铝土矿4.2吨，减少95%的空气污染。'
    },
    'PAPER': {
        'cn_name': '纸张垃圾',
        'color': '#6366f1',
        'icon': '📄',
        'description': '各类纸质废弃物',
        'examples': ['报纸', '书本', '打印纸', '杂志', '信封'],
        'disposal': ['投放到可回收物桶（蓝色）', '整理整齐后投放', '避免与厨余垃圾混合', '去除塑料封皮'],
        'tips': '废纸是重要的再生资源，回收利用可减少森林砍伐。'
    },
    'PLASTIC': {
        'cn_name': '塑料垃圾',
        'color': '#0ea5e9',
        'icon': '🧴',
        'description': '各类塑料制品废弃物',
        'examples': ['塑料瓶', '塑料袋', '塑料包装', '塑料餐具'],
        'disposal': ['投放到可回收物桶（蓝色）', '清洗并晾干后投放', '压扁塑料瓶节省空间', '区分可回收和不可回收塑料'],
        'tips': '塑料降解需要数百年，回收利用可大大减少环境污染。'
    }
}

ms.set_device("CPU")
ms.set_context(mode=0)

from mindyolo.utils.config import load_config, Config
from mindyolo.models import create_model
from mindyolo.utils.metrics import non_max_suppression, scale_coords

cfg, _, _ = load_config(CONFIG_PATH)
cfg = Config(cfg)

model = create_model(
    model_name=cfg.network.model_name,
    model_cfg=cfg.network,
    num_classes=6,
    checkpoint_path=WEIGHT_PATH
)
model.set_train(False)


def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def add_to_history(result_dict, img_path=None):
    history = load_history()
    record = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'total': len(result_dict["category_id"]),
        'categories': dict(Counter(result_dict["category_id"])),
        'scores': result_dict["score"],
        'image_path': img_path
    }
    history.append(record)
    if len(history) > 100:
        history = history[-100:]
    save_history(history)
    return history


def preprocess_img(img_bgr, img_size=IMG_SIZE, stride=32):
    h_ori, w_ori = img_bgr.shape[:2]
    r = img_size / max(h_ori, w_ori)
    if r != 1:
        interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
        img = cv2.resize(img_bgr, (int(w_ori * r), int(h_ori * r)), interpolation=interp)
    else:
        img = img_bgr.copy()
    h, w = img.shape[:2]
    if h < img_size or w < img_size:
        new_h = math.ceil(h / stride) * stride
        new_w = math.ceil(w / stride) * stride
        dh, dw = (new_h - h) / 2, (new_w - w) / 2
        top    = int(round(dh - 0.1))
        bottom = int(round(dh + 0.1))
        left   = int(round(dw - 0.1))
        right  = int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right,
                                 cv2.BORDER_CONSTANT, value=(128, 128, 128))
    img_input = img[:, :, ::-1].transpose(2, 0, 1) / 255.0
    tensor = Tensor(img_input[None], ms.float32)
    return tensor, img.shape[:2], (h_ori, w_ori)


def draw_boxes(img_bgr, result_dict):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(pil_img)

    font_size = 18
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", font_size)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

    for bbox, score, cls_id in zip(result_dict["bbox"], result_dict["score"], result_dict["category_id"]):
        x, y, w, h = bbox
        x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
        color = COLORS[cls_id % len(COLORS)]
        label = f"{CLASS_NAMES_CN[cls_id]} {score:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        try:
            bbox_text = draw.textbbox((0, 0), label, font=font)
            text_width = bbox_text[2] - bbox_text[0]
            text_height = bbox_text[3] - bbox_text[1]
        except:
            text_width = len(label) * 10
            text_height = 18
        draw.rectangle([x1, y1-text_height-6, x1+text_width+8, y1], fill=color)
        draw.text((x1+4, y1-text_height-4), label, fill=(255, 255, 255), font=font)

    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def run_detect(img_bgr):
    tensor, inp_shape, ori_shape = preprocess_img(img_bgr)
    out, _ = model(tensor)
    out = out[-1] if isinstance(out, (tuple, list)) else out
    out = out.asnumpy()
    out = non_max_suppression(out, conf_thres=CONF_THRES, iou_thres=IOU_THRES,
                              conf_free=CONF_FREE, multi_label=True, time_limit=60.0)
    result_dict = {"category_id": [], "bbox": [], "score": []}
    for pred in out:
        if len(pred) == 0:
            continue
        predn = np.copy(pred)
        scale_coords(inp_shape, predn[:, :4], ori_shape)
        for p in predn.tolist():
            x1, y1, x2, y2 = p[:4]
            result_dict["category_id"].append(int(p[5]))
            result_dict["bbox"].append([x1, y1, x2-x1, y2-y1])
            result_dict["score"].append(round(p[4], 4))
    return result_dict


def save_result_image(img_bgr):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"detect_{timestamp}.jpg"
    filepath = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(filepath, img_bgr)
    return filepath


def generate_charts(result_dict):
    chart_colors = ['#10b981', '#3b82f6', '#22c55e', '#06b6d4', '#6366f1', '#0ea5e9']

    if len(result_dict["category_id"]) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, '未检测到目标', ha='center', va='center', fontsize=16, color='#9ca3af')
        ax.axis('off')
        fig.patch.set_facecolor('#ffffff')
        return fig

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.patch.set_facecolor('#ffffff')

    class_counts = Counter(result_dict["category_id"])
    labels_cn = [CLASS_NAMES_CN[i] for i in class_counts.keys()]
    sizes = list(class_counts.values())
    colors_selected = [chart_colors[i] for i in class_counts.keys()]

    wedges, texts, autotexts = axes[0].pie(
        sizes, labels=labels_cn, colors=colors_selected, autopct='%1.1f%%',
        startangle=90, explode=[0.03]*len(sizes)
    )
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    axes[0].set_title('类别分布', fontsize=14, fontweight='bold', color='#1f2937')

    scores = result_dict["score"]
    axes[1].hist(scores, bins=10, color='#3b82f6', edgecolor='white', alpha=0.85)
    axes[1].set_xlabel('置信度', fontsize=11, color='#374151')
    axes[1].set_ylabel('数量', fontsize=11, color='#374151')
    axes[1].set_title('置信度分布', fontsize=14, fontweight='bold', color='#1f2937')
    axes[1].set_xlim(0, 1)
    axes[1].axvline(x=np.mean(scores), color='#10b981', linestyle='--', linewidth=2, label=f'均值: {np.mean(scores):.2f}')
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', alpha=0.2, color='#e5e7eb')

    plt.tight_layout()
    return fig


def detect_image(pil_input):
    if pil_input is None:
        return None, "请先上传图片", None, None

    img_bgr = cv2.cvtColor(np.array(pil_input), cv2.COLOR_RGB2BGR)
    result = run_detect(img_bgr)
    out_img = draw_boxes(img_bgr, result)

    saved_path = save_result_image(out_img)
    add_to_history(result, saved_path)
    chart_fig = generate_charts(result)

    n = len(result["category_id"])

    summary = f"检测完成\n"
    summary += f"━━━━━━━━━━━━━━━━━━━━\n"
    summary += f"检测目标数量: {n}\n"
    summary += f"保存路径: {saved_path}\n"
    summary += f"━━━━━━━━━━━━━━━━━━━━\n"

    if n > 0:
        class_counts = Counter(result["category_id"])
        summary += "\n各类别数量:\n"
        for cls_id, count in sorted(class_counts.items()):
            summary += f"  ● {CLASS_NAMES_CN[cls_id]}: {count}\n"

        scores = result["score"]
        summary += f"\n置信度统计:\n"
        summary += f"  平均: {np.mean(scores):.3f}\n"
        summary += f"  最高: {np.max(scores):.3f}\n"
        summary += f"  最低: {np.min(scores):.3f}\n"
    else:
        summary += "\n未检测到目标\n建议尝试其他图片"

    return cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB), summary, saved_path, chart_fig


def detect_video(video_path):
    if video_path is None:
        return None, "请先上传视频"

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"video_{timestamp}.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    frame_count = 0
    total_detections = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = run_detect(frame)
        frame  = draw_boxes(frame, result)
        writer.write(frame)
        frame_count += 1
        total_detections += len(result["category_id"])

    cap.release()
    writer.release()

    info = f"视频处理完成\n"
    info += f"━━━━━━━━━━━━━━━━━━━━\n"
    info += f"总帧数: {frame_count}\n"
    info += f"检测目标总数: {total_detections}\n"
    info += f"平均每帧: {total_detections/frame_count:.1f} 个\n"
    info += f"保存路径: {out_path}"

    return out_path, info


def get_statistics():
    history = load_history()

    if not history:
        return "暂无检测记录", None

    total_detections = sum(h['total'] for h in history)
    total_images = len(history)

    category_totals = Counter()
    for h in history:
        for cat_id, count in h['categories'].items():
            category_totals[int(cat_id)] += count

    stats = f"检测统计\n"
    stats += f"━━━━━━━━━━━━━━━━━━━━\n"
    stats += f"总检测次数: {total_images}\n"
    stats += f"总检测目标: {total_detections}\n"
    stats += f"平均每次: {total_detections/total_images:.1f} 个\n"
    stats += f"━━━━━━━━━━━━━━━━━━━━\n"
    stats += f"\n各类别统计:\n"
    for cls_id, count in sorted(category_totals.items()):
        pct = count / total_detections * 100
        stats += f"  ● {CLASS_NAMES_CN[cls_id]}: {count} ({pct:.1f}%)\n"

    if category_totals:
        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_facecolor('#ffffff')

        labels = [CLASS_NAMES_CN[i] for i in category_totals.keys()]
        values = list(category_totals.values())
        colors = ['#10b981', '#3b82f6', '#22c55e', '#06b6d4', '#6366f1', '#0ea5e9']
        bars_colors = [colors[i] for i in category_totals.keys()]

        bars = ax.bar(labels, values, color=bars_colors, edgecolor='white')
        ax.set_xlabel('垃圾类别', color='#374151')
        ax.set_ylabel('检测数量', color='#374151')
        ax.set_title('历史检测统计', fontsize=14, fontweight='bold', color='#1f2937')
        ax.grid(axis='y', alpha=0.2, color='#e5e7eb')

        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   str(val), ha='center', va='bottom', fontsize=10, color='#374151')

        plt.tight_layout()
        return stats, fig

    return stats, None


def list_output_files():
    files = []
    if os.path.exists(OUTPUT_DIR):
        for f in os.listdir(OUTPUT_DIR):
            if f.endswith(('.jpg', '.mp4')):
                filepath = os.path.join(OUTPUT_DIR, f)
                size = os.path.getsize(filepath)
                mtime = datetime.fromtimestamp(os.path.getmtime(filepath)).strftime("%Y-%m-%d %H:%M:%S")
                files.append([f, f"{size/1024:.1f} KB", mtime])
    return sorted(files, key=lambda x: x[2], reverse=True)


# ==================== 界面 ====================

def _guide_cards_html():
    rows = []
    for cls_name in CLASS_NAMES:
        g = GARBAGE_GUIDE[cls_name]
        tags = "".join(
            f'<span style="background:#f0f0f0;color:#555;padding:3px 10px;'
            f'border-radius:4px;font-size:11px;margin:2px;display:inline-block;">{e}</span>'
            for e in g['examples'][:4]
        )
        steps = "".join(f'<li style="margin-bottom:3px;">{d}</li>' for d in g['disposal'][:3])
        rows.append(f"""
        <div style="background:#fff;border-radius:6px;border:1px solid #e0e0e0;
                    border-left:3px solid {g['color']};padding:20px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:12px;">
                <div style="width:32px;height:32px;border-radius:6px;background:{g['color']}18;
                            display:flex;align-items:center;justify-content:center;
                            font-size:16px;flex-shrink:0;">{g['icon']}</div>
                <div>
                    <div style="font-weight:600;color:#1a1a1a;font-size:14px;">{g['cn_name']}</div>
                    <div style="color:#999;font-size:11px;">{g['description']}</div>
                </div>
            </div>
            <div style="margin-bottom:14px;">
                <div style="color:#888;font-size:11px;font-weight:600;margin-bottom:6px;">常见物品</div>
                <div>{tags}</div>
            </div>
            <div style="background:#f8f8f8;border-radius:4px;padding:12px;">
                <div style="color:#555;font-size:11px;font-weight:600;margin-bottom:6px;">投放建议</div>
                <ul style="margin:0;padding-left:16px;color:#666;font-size:12px;line-height:1.8;">{steps}</ul>
            </div>
        </div>""")
    return rows

_GUIDE_CARDS = _guide_cards_html()


def _section_title(title):
    return f"""
    <div style="padding-bottom:14px;border-bottom:1px solid #e0e0e0;margin-bottom:20px;">
        <h2 style="margin:0;font-size:18px;font-weight:600;color:#1a1a1a;">{title}</h2>
    </div>"""


CSS = """
* { box-sizing: border-box; }

.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
                 'Microsoft YaHei', 'Helvetica Neue', sans-serif !important;
    background: #f2f2f2 !important;
    padding: 0 !important;
    margin: 0 !important;
    max-width: 100% !important;
    min-height: 100vh !important;
}

footer { display: none !important; }

.main { padding: 0 !important; margin: 0 !important; }
.main > .wrap {
    max-width: 100% !important;
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
}
.gradio-container > .main,
.gradio-container > .main > .wrap,
.gradio-container > div {
    padding: 0 !important;
    margin: 0 !important;
    gap: 0 !important;
}

/* ===== 主布局: 侧边栏 + 内容区 ===== */
#main-layout {
    display: flex !important;
    flex-direction: row !important;
    gap: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
    min-height: 100vh !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    border: none !important;
    background: transparent !important;
}

/* ===== 左侧导航栏 ===== */
#sidebar {
    background: #1f1f1f !important;
    width: 192px !important;
    min-width: 192px !important;
    max-width: 192px !important;
    min-height: 100vh !important;
    padding: 0 !important;
    gap: 0 !important;
    border: none !important;
    box-shadow: none !important;
    flex-shrink: 0 !important;
    flex-grow: 0 !important;
    overflow: hidden !important;
}

#sidebar > div,
#sidebar > .block,
#sidebar .block {
    padding: 0 !important;
    margin: 0 !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
    border-radius: 0 !important;
    gap: 0 !important;
}

.sidebar-header {
    padding: 28px 20px 22px;
    color: #ffffff;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: -0.2px;
    border-bottom: 1px solid #333;
    margin-bottom: 8px;
    line-height: 1.3;
    text-align: center;
}

/* 导航按钮 - 默认 */
#sidebar button {
    background: transparent !important;
    color: #999 !important;
    border: none !important;
    border-radius: 0 !important;
    border-left: 3px solid transparent !important;
    text-align: center !important;
    justify-content: center !important;
    padding: 15px 18px !important;
    font-size: 13.5px !important;
    font-weight: 400 !important;
    box-shadow: none !important;
    width: 100% !important;
    transition: all 0.1s ease !important;
    line-height: 1.4 !important;
    margin: 0 !important;
}

#sidebar button:hover {
    background: #2a2a2a !important;
    color: #ddd !important;
}

/* 导航按钮 - 选中: 整个按钮颜色加深 */
#sidebar .active button,
#sidebar .active button:hover {
    background: #353535 !important;
    color: #ffffff !important;
    border-left: 3px solid #ffffff !important;
    font-weight: 600 !important;
}

/* ===== 右侧内容区 ===== */
#content {
    background: #fafafa !important;
    padding: 24px !important;
    min-height: 100vh !important;
    overflow-y: auto !important;
    overflow-x: hidden !important;
    flex: 1 1 0% !important;
    min-width: 0 !important;
    max-width: none !important;
    width: calc(100vw - 192px) !important;
    gap: 0 !important;
    border: none !important;
    box-shadow: none !important;
}

#content > div {
    gap: 14px !important;
    max-width: 100% !important;
    width: 100% !important;
}

/* ===== Gradio 组件全局清理 ===== */
.block {
    border: none !important;
    box-shadow: none !important;
    max-width: 100% !important;
}
.form {
    border: none !important;
    background: transparent !important;
    max-width: 100% !important;
}
.block.padded { padding: 0 !important; }
.panel { border: none !important; box-shadow: none !important; }

/* 强制所有 Gradio 容器撑满宽度 */
#content .contain,
#content .wrap,
#content .block,
#content .form,
#content .row,
#content .column {
    max-width: 100% !important;
    width: 100% !important;
}

/* ===== 操作按钮 ===== */
#content button.primary {
    background: #1a1a1a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    padding: 10px 28px !important;
    box-shadow: none !important;
    border-left: none !important;
}
#content button.primary:hover {
    background: #333 !important;
}

#content button.secondary {
    background: #fff !important;
    color: #555 !important;
    border: 1px solid #d0d0d0 !important;
    border-radius: 6px !important;
    font-weight: 500 !important;
    font-size: 14px !important;
    box-shadow: none !important;
    border-left: 1px solid #d0d0d0 !important;
}
#content button.secondary:hover {
    border-color: #999 !important;
    color: #333 !important;
    background: #f5f5f5 !important;
}

/* ===== 输入框 ===== */
.summary-text textarea {
    font-family: 'SF Mono', 'Consolas', monospace !important;
    font-size: 12.5px !important;
    line-height: 1.7 !important;
    background: #fff !important;
    color: #333 !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 6px !important;
}
.path-text textarea, .path-text input {
    background: #f5f5f5 !important;
    color: #888 !important;
    font-size: 12px !important;
    border-radius: 6px !important;
    border: 1px solid #e0e0e0 !important;
}

/* ===== 图片 ===== */
.img-upload, .vid-upload {
    border-radius: 6px !important;
    border: 1.5px dashed #ccc !important;
    background: #fff !important;
    overflow: hidden !important;
}
.img-result, .vid-result {
    border-radius: 6px !important;
    border: 1px solid #e0e0e0 !important;
    background: #fff !important;
    overflow: hidden !important;
}

/* ===== 表格 ===== */
.data-tbl table { border-collapse: collapse !important; width: 100% !important; }
.data-tbl thead th {
    background: #f5f5f5 !important;
    color: #555 !important;
    font-weight: 600 !important;
    font-size: 12.5px !important;
    padding: 12px 16px !important;
    border-bottom: 1px solid #e0e0e0 !important;
}
.data-tbl tbody td {
    padding: 10px 16px !important;
    font-size: 12.5px !important;
    color: #333 !important;
    border-bottom: 1px solid #f0f0f0 !important;
}
.data-tbl tbody tr:hover td { background: #f9f9f9 !important; }

.plot-area { border-radius: 6px !important; overflow: hidden !important; }
label span { color: #333 !important; font-weight: 500 !important; }
"""


def _nav_js(active_id):
    return (
        "() => {"
        "  document.querySelectorAll('#sidebar .nav-item').forEach(function(el) {"
        "    el.classList.remove('active');"
        "  });"
        f"  var t = document.querySelector('#{active_id}');"
        "  if (t) t.classList.add('active');"
        "}"
    )


def _make_switch(active):
    names = ["image", "video", "guide", "stats", "files"]
    return [gr.Column(visible=(n == active)) for n in names]


with gr.Blocks(title="智能垃圾分类识别系统") as demo:

    with gr.Row(elem_id="main-layout"):

        # ===== 左侧导航栏 =====
        with gr.Column(scale=0, min_width=192, elem_id="sidebar"):
            gr.HTML('<div class="sidebar-header">基于Mindspore的垃圾识别</div>')
            nav_image = gr.Button("图片检测", elem_id="nav-image", elem_classes="nav-item active")
            nav_video = gr.Button("视频检测", elem_id="nav-video", elem_classes="nav-item")
            nav_guide = gr.Button("分类指南", elem_id="nav-guide", elem_classes="nav-item")
            nav_stats = gr.Button("统计分析", elem_id="nav-stats", elem_classes="nav-item")
            nav_files = gr.Button("输出文件", elem_id="nav-files", elem_classes="nav-item")

        # ===== 右侧内容区 =====
        with gr.Column(scale=1, elem_id="content"):

            # ---- 图片检测 ----
            with gr.Column(visible=True) as p_image:
                gr.HTML(_section_title("图片检测"))
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1):
                        img_in = gr.Image(type="pil", height=360, label="上传图片", elem_classes="img-upload")
                        with gr.Row():
                            img_btn = gr.Button("开始检测", variant="primary", scale=2)
                            img_clear = gr.Button("清空", variant="secondary", scale=1)
                        saved_path = gr.Textbox(label="保存路径", interactive=False, elem_classes="path-text")
                    with gr.Column(scale=1):
                        img_out = gr.Image(height=360, label="检测结果", elem_classes="img-result")
                with gr.Row():
                    with gr.Column(scale=1):
                        summary_box = gr.Textbox(label="检测摘要", lines=8, elem_classes="summary-text")
                    with gr.Column(scale=1):
                        chart_output = gr.Plot(label="分析图表", elem_classes="plot-area")

            # ---- 视频检测 ----
            with gr.Column(visible=False) as p_video:
                gr.HTML(_section_title("视频检测"))
                with gr.Row(equal_height=True):
                    vid_in = gr.Video(label="上传视频", height=320, elem_classes="vid-upload")
                    vid_out = gr.Video(label="检测结果", height=320, elem_classes="vid-result")
                vid_btn = gr.Button("开始检测", variant="primary")
                vid_info = gr.Textbox(label="处理信息", lines=5, elem_classes="summary-text")

            # ---- 分类指南 ----
            with gr.Column(visible=False) as p_guide:
                gr.HTML(_section_title("分类指南"))
                with gr.Row():
                    for card in _GUIDE_CARDS[:3]:
                        with gr.Column(scale=1, min_width=200):
                            gr.HTML(card)
                with gr.Row():
                    for card in _GUIDE_CARDS[3:]:
                        with gr.Column(scale=1, min_width=200):
                            gr.HTML(card)

            # ---- 统计分析 ----
            with gr.Column(visible=False) as p_stats:
                gr.HTML(_section_title("统计分析"))
                refresh_stats = gr.Button("刷新统计", variant="secondary")
                with gr.Row():
                    stats_text = gr.Textbox(label="统计数据", lines=10, elem_classes="summary-text", scale=1)
                    stats_chart = gr.Plot(label="统计图表", elem_classes="plot-area", scale=1)

            # ---- 输出文件 ----
            with gr.Column(visible=False) as p_files:
                gr.HTML(_section_title("输出文件"))
                refresh_files = gr.Button("刷新文件列表", variant="secondary")
                file_list = gr.Dataframe(headers=["文件名", "大小", "修改时间"], label="", elem_classes="data-tbl")

    panels = [p_image, p_video, p_guide, p_stats, p_files]

    nav_image.click(fn=lambda: _make_switch("image"), outputs=panels, js=_nav_js("nav-image"))
    nav_video.click(fn=lambda: _make_switch("video"), outputs=panels, js=_nav_js("nav-video"))
    nav_guide.click(fn=lambda: _make_switch("guide"), outputs=panels, js=_nav_js("nav-guide"))
    nav_stats.click(fn=lambda: _make_switch("stats"), outputs=panels, js=_nav_js("nav-stats"))
    nav_files.click(fn=lambda: _make_switch("files"), outputs=panels, js=_nav_js("nav-files"))

    img_btn.click(fn=detect_image, inputs=img_in, outputs=[img_out, summary_box, saved_path, chart_output])
    img_clear.click(fn=lambda: (None, "", "", None), outputs=[img_in, summary_box, saved_path, chart_output])
    vid_btn.click(fn=detect_video, inputs=vid_in, outputs=[vid_out, vid_info])
    refresh_stats.click(fn=get_statistics, outputs=[stats_text, stats_chart])
    demo.load(fn=get_statistics, outputs=[stats_text, stats_chart])
    refresh_files.click(fn=list_output_files, outputs=file_list)
    demo.load(fn=list_output_files, outputs=file_list)


if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7864,
        share=False,
        css=CSS,
    )

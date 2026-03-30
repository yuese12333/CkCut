import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import flet as ft

from src_machine.segmenter import AutoSegmenter
from src_nn_crf.infer import CRFSegmenter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DICT_PRIMARY = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_primary.txt")
DICT_WIKI = os.path.join(BASE_DIR, "data", "output_dict", "my_dict_wiki.txt")
HMM_MODEL_PATH = os.path.join(BASE_DIR, "data", "output_dict", "hmm_model.json")

DL_MODELS: Dict[str, str] = {
    "综合模型 (Merged)": os.path.join(BASE_DIR, "data", "output_nn_crf_merged", "bilstm_crf.pt"),
    "单模型 as_train": os.path.join(BASE_DIR, "data", "output_nn_crf_single", "as_train", "bilstm_crf.pt"),
    "单模型 cityu_train": os.path.join(BASE_DIR, "data", "output_nn_crf_single", "cityu_train", "bilstm_crf.pt"),
    "单模型 msr_training": os.path.join(BASE_DIR, "data", "output_nn_crf_single", "msr_training", "bilstm_crf.pt"),
    "单模型 pku_training": os.path.join(BASE_DIR, "data", "output_nn_crf_single", "pku_training", "bilstm_crf.pt"),
}

LARGE_CHAR_LIMIT = 10000
loaded_models: Dict[str, Any] = {}


@dataclass
class TaskItem:
    task_type: str
    config: Any
    name: str


def guess_crf_dims(model_path: str) -> Tuple[int, int]:
    default_dims = (128, 256)
    history_path = os.path.join(os.path.dirname(model_path), "train_history.json")
    if not os.path.exists(history_path):
        return default_dims
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        return int(cfg.get("embedding_dim", default_dims[0])), int(cfg.get("hidden_dim", default_dims[1]))
    except Exception:
        return default_dims


def get_segmenter(model_type: str, config: Any):
    cache_key = f"{model_type}|{config}"
    if cache_key in loaded_models:
        return loaded_models[cache_key]

    if model_type == "mechanical":
        dict_path, use_hmm = config
        hmm_path = HMM_MODEL_PATH if use_hmm else None
        model = AutoSegmenter(dict_path, hmm_model_path=hmm_path)
        if not use_hmm:
            model.hmm_enabled = False
    else:
        model_path = str(config)
        vocab_path = os.path.join(os.path.dirname(model_path), "char_vocab.json")
        emb_dim, hidden_dim = guess_crf_dims(model_path)
        model = CRFSegmenter(
            model_path=model_path,
            vocab_path=vocab_path,
            embedding_dim=emb_dim,
            hidden_dim=hidden_dim,
            device="auto",
        )

    loaded_models[cache_key] = model
    return model


def main(page: ft.Page):
    page.title = "CkCut 分词器工作台"
    page.theme_mode = ft.ThemeMode.LIGHT
    page.theme = ft.Theme(font_family="Microsoft YaHei UI")
    page.dark_theme = ft.Theme(font_family="Microsoft YaHei UI")
    page.window_width = 1200
    page.window_height = 850
    page.padding = 16
    page.scroll = ft.ScrollMode.AUTO

    current_input_text = ""
    is_large_file = False
    active_tasks: List[TaskItem] = []
    last_output_data: Dict[str, str] = {}

    input_textfield = ft.TextField(
        label="输入待分词文本（或导入 TXT）",
        multiline=True,
        min_lines=5,
        max_lines=10,
        expand=True,
        text_style=ft.TextStyle(font_family="Microsoft YaHei UI", size=14),
    )
    file_info_text = ft.Text("当前未导入文件", color=ft.Colors.GREY_600)
    status_text = ft.Text("就绪", color=ft.Colors.BLUE_GREY_700)

    mech_dict_dropdown = ft.Dropdown(
        label="选择机械分词词典",
        options=[
            ft.dropdown.Option(key=DICT_PRIMARY, text="my_dict_primary.txt"),
            ft.dropdown.Option(key=DICT_WIKI, text="my_dict_wiki.txt"),
        ],
        value=DICT_PRIMARY,
        width=260,
    )
    mech_hmm_switch = ft.Switch(label="启用 HMM 未登录词识别", value=True)

    dl_model_dropdown = ft.Dropdown(
        label="选择深度学习 PT 模型",
        options=[ft.dropdown.Option(key=v, text=k) for k, v in DL_MODELS.items()],
        value=list(DL_MODELS.values())[0],
        width=350,
    )

    active_tasks_listview = ft.ListView(expand=True, spacing=8, height=170)
    results_column = ft.Column(scroll=ft.ScrollMode.AUTO, expand=True, spacing=12)

    run_button = ft.FilledButton("运行所选模型", icon=ft.Icons.PLAY_ARROW, width=220, height=48)

    mech_panel = ft.Card(
        content=ft.Container(
            padding=12,
            content=ft.Column(
                [
                    ft.Text("机械分词配置", weight=ft.FontWeight.BOLD),
                    mech_dict_dropdown,
                    mech_hmm_switch,
                    ft.Button("加入对比队列", icon=ft.Icons.ADD, on_click=lambda _: add_mech_task()),
                ]
            ),
        )
    )

    dl_panel = ft.Card(
        content=ft.Container(
            padding=12,
            content=ft.Column(
                [
                    ft.Text("深度学习分词配置", weight=ft.FontWeight.BOLD),
                    dl_model_dropdown,
                    ft.Container(height=32),
                    ft.Button("加入对比队列", icon=ft.Icons.ADD, on_click=lambda _: add_dl_task()),
                ]
            ),
        )
    )

    def show_snack(msg: str):
        page.snack_bar = ft.SnackBar(ft.Text(msg))
        page.snack_bar.open = True
        page.update()

    def update_task_list():
        active_tasks_listview.controls.clear()
        for i, task in enumerate(active_tasks):
            active_tasks_listview.controls.append(
                ft.ListTile(
                    leading=ft.Icon(ft.Icons.SETTINGS if task.task_type == "mechanical" else ft.Icons.MEMORY),
                    title=ft.Text(task.name),
                    trailing=ft.IconButton(
                        icon=ft.Icons.DELETE,
                        icon_color=ft.Colors.RED_600,
                        on_click=lambda _, idx=i: remove_task(idx),
                    ),
                )
            )
        page.update()

    def add_mech_task():
        config = (mech_dict_dropdown.value, bool(mech_hmm_switch.value))
        for t in active_tasks:
            if t.task_type == "mechanical" and t.config == config:
                show_snack("该机械分词配置已在队列中，不能重复添加")
                return

        dict_name = "my_dict_primary.txt" if mech_dict_dropdown.value == DICT_PRIMARY else "my_dict_wiki.txt"
        hmm_status = "开启HMM" if mech_hmm_switch.value else "关闭HMM"
        task = TaskItem("mechanical", config, f"机械分词 [{dict_name} | {hmm_status}]")
        active_tasks.append(task)
        update_task_list()
        show_snack("已加入机械分词配置")

    def add_dl_task():
        model_path = str(dl_model_dropdown.value)
        for t in active_tasks:
            if t.task_type == "dl" and t.config == model_path:
                show_snack("该深度学习模型已在队列中，不能重复添加")
                return

        model_name = next((k for k, v in DL_MODELS.items() if v == model_path), "深度模型")
        task = TaskItem("dl", model_path, f"深度学习 [{model_name}]")
        active_tasks.append(task)
        update_task_list()
        show_snack("已加入深度学习配置")

    def remove_task(index: int):
        if 0 <= index < len(active_tasks):
            active_tasks.pop(index)
            update_task_list()

    def run_for_text(text: str, export_path: str | None = None):
        nonlocal last_output_data
        if not text.strip():
            show_snack("输入内容为空")
            return
        if not active_tasks:
            show_snack("请至少加入一个分词配置")
            return
        if len(text) > LARGE_CHAR_LIMIT and not export_path:
            show_snack("文本过大，请使用【运行并导出到TXT】")
            return

        run_button.disabled = True
        run_button.text = "运行中..."
        page.update()
        try:
            results_column.controls.clear()
            page.update()
            output_data: Dict[str, str] = {}

            for task in active_tasks:
                try:
                    segmenter = get_segmenter(task.task_type, task.config)
                    start = time.time()
                    words = segmenter.cut(text)
                    elapsed = time.time() - start
                    result_str = " / ".join(words)
                    output_data[task.name] = result_str

                    if not export_path:
                        results_column.controls.append(
                            ft.Card(
                                content=ft.Container(
                                    padding=12,
                                    content=ft.Column(
                                        [
                                            ft.Row(
                                                [
                                                    ft.Text(task.name, weight=ft.FontWeight.BOLD),
                                                    ft.Text(f"耗时: {elapsed:.4f}s", color=ft.Colors.GREY_600),
                                                ],
                                                alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                                            ),
                                            ft.Divider(),
                                            ft.Text(result_str, selectable=True, font_family="Microsoft YaHei UI", size=14),
                                        ]
                                    ),
                                )
                            )
                        )
                except Exception as ex:
                    results_column.controls.append(ft.Text(f"{task.name} 运行失败: {ex}", color=ft.Colors.RED_700))

            if export_path:
                try:
                    with open(export_path, "w", encoding="utf-8") as f:
                        for name, res in output_data.items():
                            f.write(f"=== {name} ===\n")
                            f.write(res + "\n\n")
                    show_snack("分词结果导出成功")
                except Exception as ex:
                    show_snack(f"导出失败: {ex}")
            else:
                show_snack("分词完成")

            last_output_data = output_data
            page.update()
        finally:
            run_button.disabled = False
            run_button.text = "运行所选模型"
            page.update()

    async def handle_import_click(_=None):
        nonlocal current_input_text, is_large_file
        files = await import_picker.pick_files(allow_multiple=False, allowed_extensions=["txt"])
        if not files:
            return
        filepath = files[0].path
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                current_input_text = f.read()
            file_info_text.value = f"已导入: {os.path.basename(filepath)}"

            if len(current_input_text) > LARGE_CHAR_LIMIT:
                is_large_file = True
                input_textfield.value = f"【文件过大 ({len(current_input_text)} 字)】已加载到内存。请使用【运行并导出到 TXT】。"
                input_textfield.disabled = True
            else:
                is_large_file = False
                input_textfield.value = current_input_text
                input_textfield.disabled = False

            status_text.value = "导入成功"
            status_text.color = ft.Colors.GREEN_700
        except Exception as ex:
            status_text.value = f"导入失败: {ex}"
            status_text.color = ft.Colors.RED_700
        page.update()

    async def handle_export_click(_=None):
        out_path = await export_picker.save_file(allowed_extensions=["txt"])
        if not out_path:
            return
        try:
            text_to_cut = current_input_text if is_large_file else (input_textfield.value or "")
            run_for_text(text_to_cut, export_path=out_path)
            status_text.value = f"已导出: {out_path}"
            status_text.color = ft.Colors.GREEN_700
        except Exception as ex:
            status_text.value = f"导出失败: {ex}"
            status_text.color = ft.Colors.RED_700
        page.update()

    def clear_input(_=None):
        nonlocal current_input_text, is_large_file
        current_input_text = ""
        is_large_file = False
        input_textfield.disabled = False
        input_textfield.value = ""
        file_info_text.value = "当前未导入文件"
        page.update()

    def clear_output(_=None):
        nonlocal last_output_data
        last_output_data = {}
        results_column.controls.clear()
        status_text.value = "输出已清空"
        status_text.color = ft.Colors.BLUE_GREY_700
        page.update()

    import_picker = ft.FilePicker()
    export_picker = ft.FilePicker()
    page.services.extend([import_picker, export_picker])

    page.add(
        ft.Text("CkCut 分词引擎对比工作台", size=24, weight=ft.FontWeight.BOLD),
        ft.Row([
            ft.Button("从 TXT 导入文本", icon=ft.Icons.UPLOAD_FILE, on_click=handle_import_click),
            ft.OutlinedButton("清空输入", on_click=clear_input),
            ft.OutlinedButton("清空输出", on_click=clear_output),
        ], wrap=True),
        ft.Row([
            input_textfield,
            ft.Container(content=file_info_text, width=320, padding=8),
        ], alignment=ft.MainAxisAlignment.START),
        ft.Divider(),
        ft.Text("添加对比配置", size=18, weight=ft.FontWeight.BOLD),
        ft.Row([
            ft.Container(content=mech_panel, expand=1),
            ft.Container(content=dl_panel, expand=1),
        ], spacing=12, vertical_alignment=ft.CrossAxisAlignment.START),
        ft.Row([
            ft.Container(
                width=700,
                border=ft.Border.all(1, ft.Colors.OUTLINE),
                border_radius=6,
                padding=10,
                content=ft.Column([
                    ft.Text("当前启用的模型队列（运行时会同时执行）", color=ft.Colors.GREY_700),
                    active_tasks_listview,
                ]),
            ),
            ft.Column([
                run_button,
                ft.FilledTonalButton(
                    "运行并导出到 TXT",
                    icon=ft.Icons.SAVE,
                    width=220,
                    height=48,
                    on_click=handle_export_click,
                ),
            ], alignment=ft.MainAxisAlignment.CENTER),
        ], vertical_alignment=ft.CrossAxisAlignment.CENTER),
        ft.Divider(),
        ft.Text("分词结果", size=18, weight=ft.FontWeight.BOLD),
        results_column,
        status_text,
    )

    run_button.on_click = lambda _: run_for_text(current_input_text if is_large_file else (input_textfield.value or ""))


if __name__ == "__main__":
    ft.run(main)

# depth_warp_vs/ui_app.py
import sys

def main():
    try:
        # 优先包内导入
        from depth_warp_vs.ui.app import main as app_main
    except Exception:
        # 兼容直接从源码根目录运行
        from ui.app import main as app_main
    app_main()

if __name__ == "__main__":
    # 兼容 python depth_warp_vs/ui_app.py 直接执行
    # 将源码根目录加入 sys.path，便于从源树运行
    import os
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if root not in sys.path:
        sys.path.insert(0, root)
    main()

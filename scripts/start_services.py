# scripts/start_services.py
#!/usr/bin/env python3
"""
一鍵啟動所有服務
"""
import subprocess
import time
import sys
import signal
from pathlib import Path
import threading


class ServiceManager:
    def __init__(self):
        self.processes = []
        self.running = False

    def start_service(self, name, command, cwd=None, env=None):
        """Start a service in background"""
        print(f"🚀 啟動 {name}...")

        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            self.processes.append(
                {"name": name, "process": process, "command": command}
            )

            # Monitor process in background
            threading.Thread(
                target=self._monitor_process, args=(name, process), daemon=True
            ).start()

            return True

        except Exception as e:
            print(f"❌ 啟動 {name} 失敗: {e}")
            return False

    def _monitor_process(self, name, process):
        """Monitor process output"""
        while process.poll() is None:
            try:
                output = process.stdout.readline()
                if output:
                    print(f"[{name}] {output.strip()}")
            except:
                break

    def stop_all(self):
        """Stop all services"""
        print("\n🛑 停止所有服務...")

        for service in self.processes:
            try:
                service["process"].terminate()
                print(f"✅ {service['name']} 已停止")
            except:
                try:
                    service["process"].kill()
                except:
                    pass

        self.processes.clear()
        self.running = False

    def start_all(self):
        """Start all VisionQuest services"""
        self.running = True

        # Backend API
        if not self.start_service(
            "Backend API",
            "uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000",
        ):
            return False

        # Wait for backend to start
        print("⏳ 等待後端啟動...")
        time.sleep(3)

        # Gradio UI
        self.start_service("Gradio UI", "python app.py", cwd="frontend/gradio_app")

        # React UI (if package.json exists)
        react_path = Path("frontend/react_app/package.json")
        if react_path.exists():
            self.start_service("React UI", "npm run dev", cwd="frontend/react_app")

        return True


def signal_handler(sig, frame):
    """Handle Ctrl+C"""
    print("\n⚠️  收到中斷信號...")
    if hasattr(signal_handler, "manager"):
        signal_handler.manager.stop_all()
    sys.exit(0)


def main():
    # Setup signal handler
    manager = ServiceManager()
    signal_handler.manager = manager
    signal.signal(signal.SIGINT, signal_handler)

    print("🎯 VisionQuest 服務啟動器")
    print("=" * 40)

    if not manager.start_all():
        print("❌ 啟動失敗")
        return

    print("\n✅ 所有服務已啟動!")
    print("\n📱 服務地址:")
    print("   Backend API: http://localhost:8000")
    print("   API 文檔: http://localhost:8000/docs")
    print("   Gradio UI: http://localhost:7860")
    print("   React UI: http://localhost:3000")
    print("\n按 Ctrl+C 停止所有服務")

    # Keep running
    try:
        while manager.running:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        manager.stop_all()


if __name__ == "__main__":
    main()

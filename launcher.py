import tkinter as tk
import subprocess
import sys

def run_and_return(script_name):
    root.destroy()  # close the launcher window

    # Run the external script and wait until it finishes
    subprocess.call([sys.executable, script_name])

    # When script finishes, reopen the launcher
    main()

def main():
    global root
    root = tk.Tk()
    root.title("Sentinel-2 Launcher")

    label = tk.Label(root, text="Do you want to download new products or fuse existing ones?")
    label.pack(pady=20)

    download_button = tk.Button(root, text="Download new products",
                                command=lambda: run_and_return("src/downloader.py"),
                                width=30, height=2)
    download_button.pack(pady=10)

    fuse_button = tk.Button(root, text="Fuse products",
                            command=lambda: run_and_return("src/fuser.py"),
                            width=30, height=2)
    fuse_button.pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    main()


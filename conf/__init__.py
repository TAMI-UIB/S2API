from dotenv import load_dotenv
import os

os.environ["PROJECT_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

load_dotenv(os.path.join(os.environ["PROJECT_ROOT"], ".env"))
print("hola")
print(os.environ["PROJECT_ROOT"])
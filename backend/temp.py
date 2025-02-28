from utils import Utils
from dotenv import load_dotenv
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--class_name', type=str, help='Name of the class')


args = parser.parse_args()
class_name = args.class_name
load_dotenv('.env')

uti = Utils()

uti.save_to_s3(f"E:\\my_one\\media\\videos\\{class_name}\\480p15\\{class_name}.mp4",f"{class_name}.mp4","video/mp4")


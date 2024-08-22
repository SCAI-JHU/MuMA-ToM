import cv2
import os
import ipdb
from pathlib import Path
import re
import json
import datetime
from moviepy.editor import ImageClip, concatenate_videoclips, TextClip, CompositeVideoClip, VideoFileClip, clips_array
from moviepy.config import change_settings
from PIL import Image, ImageDraw, ImageFont
import textwrap
from overlay_generation.generate_overlay import overlay_text_on_image
from render_new import get_chars
import pickle
import argparse
import random
os.environ["MAGICK_CONFIGURE_PATH"] = "/home/scai/Workspace/hshi33/virtualhome/data"
directory = "/home/scai/Workspace_2/hshi33/video_color"
parser = argparse.ArgumentParser(description="Collection data simulator.")
parser.add_argument("--id", type=int, default=0)
args = parser.parse_args()
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})
json_file = open("record.json", "r")
log_data = json.load(json_file)
caption = True
def extract_episode_number(file_name):
    pattern = r'logs_episode\.(\d+)_iter\.\d+\.pik'
    
    match = re.search(pattern, file_name)
    
    if match:
        return match.group(1)
    else:
        return None
def create_video_from_images(episode_id):
    with open(os.path.join(directory, str(episode_id), "frame_intervals.pik"), "rb") as file:
        intervals = pickle.load(file)
    with open(os.path.join(directory, str(episode_id), "timemark.json"), "r") as file:
        frame_change = json.load(file)["time_change_frame"]
    for index, interval in enumerate(intervals):
        if interval[1] == frame_change:
            frame_change = intervals[index+1][1]
            break
    file_list = os.listdir(os.path.join(directory, str(episode_id)))
    file_list.sort()
    all_images = []
    for file in file_list:
        if file.endswith("pik") or file.endswith("json") or file.endswith("mp4"):
            continue
        images = [img for img in os.listdir(os.path.join(directory, str(episode_id), file, "0")) if img.endswith("normal.png")]
        images.sort()
        for image in images:
            all_images.append(os.path.join(directory, str(episode_id), file, "0", image))
             
    perspective_change = False
    count = 0
    video_path = "/home/scai/Workspace_2/hshi33/benchmark/raw_video_frames/{}/video_before.mp4".format(episode_id)
    output_images = []
    for image in all_images:
        if count > frame_change and not perspective_change:
            clips = [ImageClip(img, duration=0.1) for img in output_images]
            final_clip = concatenate_videoclips(clips, method="compose")
            final_clip.write_videofile(video_path, fps=20)
            output_images = []
            video_path = video_path.replace("before", "{}".format(episode_id))
            perspective_change = True
        output_images.append(os.path.join(image_folder, image))
        count += 1
    clips = [ImageClip(img, duration=0.1) for img in output_images]
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(video_path, fps=20)

'''    for image in images:
        os.remove(os.path.join(image_folder, image))'''
def modify_language(language):
    objects = ["book", "remotecontrol", "potato", "carrot", "bread", "milk", "wineglass",
        "cellphone", "toy", "spoon", "mug", "juice", "beer", "wine", "folder", "magazine", 
        "coffeepot", "glass", "notes", "check", "address_book", "wine glass", "remote control"]
    word = None
    for obj in objects:
        if obj in language:
            word = obj
    if obj is None:
        ipdb.set_trace()
    index1 = language.find("inside")
    index2 = language.find("on")
    if index1 == -1:
        pattern = re.search(r'\b(on\s+.+)', language)
    elif index2 == -1:
        pattern = re.search(r'\b(inside\s+.+)', language)
    elif index1 < index2:
        pattern = re.search(r'\b(inside\s+.+)', language)
    elif index2 < index1:
        pattern = re.search(r'\b(on\s+.+)', language)
    pattern = re.search(r'\b(inside\s+.+)', language)
    phrase = pattern.group(1) 
    potential_templates = [f"I discovered a {word} {phrase}", f"There was a {word} {phrase}", f"I found a {word} {phrase}", f"I spotted a {word} {phrase}", f"I located a {word} {phrase}"]

    return potential_templates[3]

def create_video_from_images_language(episode_id):
    with open(os.path.join(directory, str(episode_id), "frame_intervals.pik"), "rb") as file:
        intervals = pickle.load(file)
    text = []
    with open(os.path.join(directory, str(episode_id), "language.json"), "r") as file:
        data = json.load(file)
        frame_change = data["time_change_frame"]
        text.append(data["0"])
        text.append(modify_language(data["1"]))
    if frame_change == 0:
        frame_change = intervals[0][1]
    else:
        for index, interval in enumerate(intervals):
            if interval[1] == frame_change:
                frame_change = intervals[index][1]
                break
    file_list = os.listdir(os.path.join(directory, str(episode_id)))
    file_list.sort()
    all_images = []
    for file in file_list:
        if file.endswith("pik") or file.endswith("json") or file.endswith("mp4") or file == "0":
            continue
        images = [img for img in os.listdir(os.path.join(directory, str(episode_id), file, "0")) if img.endswith("normal.png")]
        images.sort()
        for image in images:
            all_images.append(os.path.join(directory, str(episode_id), file, "0", image))

    male_names = ["John", "Michael", "David", "Kevin"]
    female_names = ["Mary", "Sarah", "Jessica", "Emma"]

    male_chars = ["Chars/Male1", "Chars/Male2", "Chars/Male1", "Chars/Male6"]
    female_chars = ["Chars/Female1", "Chars/Female2", "Chars/Female4", "Chars/Female4"]

    male_char, female_char = get_chars(episode_id)
    male_name = male_names[male_chars.index(male_char)]
    female_name = female_names[female_chars.index(female_char)]
    male_char += ".png"
    female_char += ".png"
    avators=[female_char, male_char]

    text[1] = male_name + ": " + text[1]
    text[0] = female_name + ": " + text[0]
    print(text[0])
    print(text[1])
    text_info = text[0] + "\n" + text[1]
    with open("/home/scai/Workspace_2/hshi33/benchmark/texts/multimodal/episode_{}.txt".format(episode_id), "w") as text_file:
        text_file.write(text_info)
    images_and_captions = []
    language_record = False
    count = 0
    for image in all_images:
        if count > frame_change and not language_record:
            language_record = True
            images_and_captions.append((image, text))
        else:
            images_and_captions.append((image, None))
            count += 1
    
    output_images = []
    output_images2 = []
    first_language = False
    for index, (image_path, caption) in enumerate(images_and_captions):
        # Create an ImageClip for the image
        if caption and not first_language:
            output_image_path = f"temp_image_{index}.jpg"
            overlay_text_on_image(image_path, text, output_image_path, "overlay_generation/Lora-VariableFont_wght.ttf", 50, avators)
            for i in range(50):
                output_images.append(output_image_path)
            first_language = True
            output_images2.append(image_path)
        else:
            output_images.append(image_path)
            output_images2.append(image_path)

    clips = [ImageClip(img, duration=0.1) for img in output_images]

    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(os.path.join(directory, str(episode_id), "video_{}.mp4".format(episode_id)), fps=20)

    for img in output_images:
        if img.startswith("temp_image_"):
            try:
                os.remove(img)
            except FileNotFoundError:
                continue

episode_list = [5302, 5379, 5381, 5082, 5175, 5039, 4043, 4198, 4525, 4063, 4057, 4034, 4441, 4527, 4567, 4575, 4117, 4077, 5042, 4331, 4529, 4101, 4103, 4488, 4102, 4540, 4372, 5197, 4059, 4526, 4145, 4487, 5091, 4482, 4133, 4017, 4416, 4081, 4083, 5163, 4369, 5090, 4485, 4449, 4551, 4623, 4559, 4200, 4105, 5017, 4365, 5084, 5127, 4037, 4473, 4113, 4385, 4312, 4499, 4023, 5173, 4463, 4184, 4047, 4118, 4123, 4334, 4328, 4106, 4419, 4667, 4641, 4455, 4343, 4166, 4098, 5121, 4490, 4339, 4190, 5103, 4164, 4370, 4172, 4423, 4009, 5176, 4005, 4224, 4576, 5165, 4542, 4018, 4162, 4453, 4054, 4520, 4329, 5105, 4546, 5010, 4332, 4452, 4658, 5154, 4556, 4327, 4560, 4656, 4505, 4439, 4512, 5095, 4173, 5068, 4367, 4262, 4284, 4324, 5126, 5049, 4594, 4150, 4285, 5138, 5184, 4429, 4568, 4338, 4458, 5080, 4041, 4469, 4621, 4124, 4604, 5099, 5014, 5098, 4355, 4070, 4176, 4606, 4178, 4552, 4584, 4618, 4506, 4160, 5093, 4374, 5509, 4510]
language_list = [5302, 5379, 5381, 5082, 5175, 5039, 4043, 4198, 4525, 4063, 4057, 4034, 4441, 4527, 4567, 4575, 4117, 4077, 5042, 4331, 4529, 4101, 4103, 4488, 4102, 4540, 4372, 5197, 4059, 4526, 4145, 4487, 5091, 4482, 4133, 4017, 4416, 4081, 4083, 5163, 4369, 5090, 4485, 4449, 4551, 4623, 4559, 4200, 4105, 5017, 4365, 5084, 5127, 4037, 4473, 4113, 4385, 4312, 4499, 4023, 5173, 4463, 4184, 4047, 4118, 4123, 4334, 4328, 4106, 4419, 4667, 4641, 4455, 4343, 4166, 4098, 5121, 4490, 4339, 4190, 5103, 4164, 4370, 4172, 4423, 4009, 5176, 4005, 4224, 4576, 5165, 4542, 4018, 4162, 4453, 4054, 4520, 4329, 5105, 4546, 5010, 4332, 4452, 4658, 5154, 4556, 4327, 4560, 4656, 4505, 4439, 4512, 5095, 4173, 5068, 4367, 4262, 4284, 4324, 5126, 5049, 4594, 4150, 4285, 5138, 5184, 4429, 4568, 4338, 4458, 5080, 4041, 4469, 4621, 4124, 4604, 5099, 5014, 5098, 4355, 4070, 4176, 4606, 4178, 4552, 4584, 4618, 4506, 4160, 5093, 4374, 5509, 4510]
nolang_list = [766, 2709, 404, 2726, 640, 549, 609, 538, 3289, 135, 2053, 801, 857, 144, 3082, 138, 642, 3068, 153, 1827, 3355, 263, 647, 557, 784, 3308, 913, 130, 634, 3058, 644, 195, 895, 3092, 541, 865, 389, 128, 154, 628, 3117, 2780, 223, 3129, 1883, 212, 2070, 858, 393, 1856, 3292, 578, 2559, 1802, 548, 225, 910, 2799, 42, 193, 2462, 3366, 1758, 161, 1818, 3300, 397, 630, 956, 3130, 3315, 612, 3098, 3074, 3166, 905, 3119, 179, 1131, 129, 1817, 871, 532, 583, 1819, 152, 655, 824, 1811, 3077, 577, 790, 1858, 848, 3050, 3060, 528, 682, 1765, 601, 1893]
episode_list = nolang_list
episode_list = [args.id]
files = []
failed = []
too_long = []
for episode_id in episode_list:
    files.append(str(episode_id))
for file in files:
    if int(file) < 4000:
        directory = "/home/scai/Workspace_2/hshi33/video_new"
    else:
        directory = "/home/scai/Workspace_2/hshi33/benchmark/raw_video_frames"
    print(directory)
    '''try:'''
    if not Path(os.path.join(directory, file)).is_dir():
        continue
    render = True
    if not render:
        continue
    '''if Path(os.path.join(directory, file, 'video_{}.mp4'.format(file))).is_file():
        print("video of episode {} was produced".format(file))
        continue
    if Path(os.path.join(directory, file, 'character0.mp4')).is_file():
        print("video of episode {} was produced".format(file))
        continue
    if Path(os.path.join(directory, file, "episode_{}.mp4".format(file))).is_file():
        print("video of episode {} was produced".format(file))
        continue'''
    print("Producing video of episode {}".format(file))
    if Path(os.path.join(directory, file, "language.json")).is_file() and caption:

        create_video_from_images_language(int(file))

    elif Path(os.path.join(directory, file, "timemark.json")).is_file():
        image_folder = os.path.join(directory, file, "0") 
        video_path_0 = os.path.join(directory, file, 'character0.mp4')
        with open(os.path.join(directory, file, "timemark.json"), "r") as time_file:
            data = json.load(time_file)
        change_frame = data["time_change_frame"]
        fps = 20

        create_video_from_images(episode_id)

    else:
        print("problem in rendering procedure for episode {}".format(file))
        failed.append(file)
        continue
    '''except Exception as e:
        
        print(str(e))
        print("Failed files", file)
        failed.append(file)
        continue'''
print("too long: ", too_long)
print("failed in rendering: ", failed)

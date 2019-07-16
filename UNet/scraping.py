from pytube import YouTube
from pydub import AudioSegment
import csv
import json
cnt = 0
def audio_download(row):
    global ClassList_, Class_num, cnt
    cnt += 1
    for row_i in row[3:]:
        try:
            Class_num[row_i] += 1
        except KeyError:
            continue
        print("Count up to {}:{}".format(row_i, Class_num[row_i]))
        #だいたい200個のテストデータを取得
        if Class_num[row_i] > int(200 / len(ClassList_)):
            print("{}:Reach the upper limit. Ignore.".format(row_i))
            return 0
    try:
        url = "http://youtu.be/{}".format(row[0])
        yt = YouTube(url)
        yt.streams.get_by_itag(140).download("./training_data/youtube/", str(cnt))
        mp4 = AudioSegment.from_file("./training_data/youtube/{}".format(str(cnt)) + ".mp4", "mp4")
        triming_mp4 = mp4[int(float(row[1]))*1000:int(float(row[2]))*1000]
        triming_mp4.export("./Noise/{}".format(str(cnt)) + ".wav", format="wav")
        return 0
    except:
        print("This video is unavailable. Ignore.")
        return 0
    

f = open("ontology.json", 'r')
json_dict = json.load(f)
#ほしいontologyを書いとく(https://research.google.com/audioset/ontology/index.html)
ClassList_ = [
                "Human voice",
                "Whistling",
                "Respiratory sounds",
                "Human locomotion",
                "Digestive",
                "Hands",
                "Heart sounds, heartbeat",
                "Otoacoustic emission",
                "Human group actions",
                "Sound reproduction",
                "Noise",
                "Acoustic environment",
                "Wind",
                "Vehicle",
                "Domestic sounds, home sounds",
                "Bell",
                "Mechanisms",
                "Alarm"
            ] 
ClassList = []
for i in range(len(json_dict)):
    if json_dict[i]["name"] in ClassList_ :
        ClassList.append(json_dict[i]["id"])
        print("id:{},name:{}".format(json_dict[i]["id"], json_dict[i]["name"]))
Class_num = {str(cls_):0 for cls_ in ClassList}
print(Class_num)
#len(row) >= 3, row = [YTID,START,END,Class1,Class2...]

with open('balanced_train_segments.csv', 'r') as f:
    reader = csv.reader(f)
    for row_ in reader:
        row = [i.strip().strip('"') for i in row_]
        if r'#' not in row[0]:
            if len(set(row[3:]) & set(ClassList)) != 0:
                print(row)
                _ = audio_download(row)
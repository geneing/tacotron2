import os
import glob
import numpy as np
import argparse
import nltk.data
from aeneas.executetask import ExecuteTask
from aeneas.task import Task

from pydub import AudioSegment
import json


from nltk.tokenize import sent_tokenize

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str,
                    required=True, help='path to the directory containing sound files')
args = parser.parse_args()


text_dir = args.input_dir+'/txt/'

text_out_dir = args.input_dir+'/clean_txt/'
os.makedirs(text_out_dir, exist_ok=True)
audio_in_dir = args.input_dir+'/'

align_out_dir = args.input_dir+'/align/'
os.makedirs(align_out_dir, exist_ok=True)


text_list=sorted(glob.glob(text_dir+'/*.txt'))
audio_list=sorted(glob.glob(audio_in_dir+'/*.mp3'))

if 0:
    # create Task object
    config_string = u"task_language=eng|is_text_type=plain|os_task_file_format=json"

    #text_file = 'ourmutualfriend_01_dickens_64kb.txt'
    for text,audio in zip(text_list, audio_list):
        # load text from 1st chapter
        fname = os.path.basename(text)
        with open(text, 'r') as f:
            data = f.read()
            paragraphs = data.split('\n\n')
            paragraph_sentence_list = []
            for paragraph in paragraphs:
                paragraph = paragraph.replace('\n', ' ')
                paragraph_sentence_list += tokenizer.tokenize(paragraph)


        with open(text_out_dir+fname,'w') as f:
            for psl in paragraph_sentence_list:
                f.write('%s\n'%psl)


        task = Task(config_string=config_string)
        task.audio_file_path_absolute = audio
        task.text_file_path_absolute = text_out_dir+fname
        task.sync_map_file_path_absolute = align_out_dir+"/%s"%fname[:-4]+'.json'
        ExecuteTask(task).execute()
        task.output_sync_map_file()




segmented_out_dir = args.input_dir+'/segmented/'
os.makedirs(segmented_out_dir, exist_ok=True)
syncmap_list = sorted(glob.glob(align_out_dir+'/*.json'))

filelist_fout = open(segmented_out_dir+'filelist.txt','w')

for text, audio, syncmap_file in zip(text_list, audio_list, syncmap_list):

    book = AudioSegment.from_mp3(audio)

    with open(syncmap_file) as f:
        syncmap = json.loads(f.read())

    sentences = []
    for fragment in syncmap['fragments']:
        if((float(fragment['end'])*1000)-float(fragment['begin'])*1000) > 400:
            if( float(fragment['begin']) > 30 ): #skip the beginning, it's often librivox disclaimer
                sentences.append({"audio":book[float(fragment['begin'])*1000:float(fragment['end'])*1000], "text":fragment['lines'][0]})


    fname_prefix=os.path.basename(audio)[:-4]
    for idx, sentence in enumerate(sentences):
        sentence['audio'].export(segmented_out_dir+'/'+fname_prefix+'_%.3d'%idx+'.wav', format='wav')
        filelist_fout.write('DUMMY/%s|%s\n'%(fname_prefix+'_%.3d'%idx+'.wav', sentence['text']))
    filelist_fout.flush()


print('end')


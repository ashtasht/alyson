import os
import multiprocessing
import re
import datetime
import tempfile
import numpy as np
from collections import Counter
from tokenizers import BertWordPieceTokenizer

omitted_strings = ["<media omitted>", "messages to this chat and calls are now secured with end-to-end encryption. tap for more info.", "you deleted this message", "this message was deleted", "missed voice call"]

class Message:
    def __init__(self, time, sender, content, i, chat_id):
        self.content = content
        self.time = time
        self.sender = sender
        self.i = i
        self.chat_id = chat_id

    def __str__(self):
        if self.time > 2<<16:
            dt = datetime.datetime.fromtimestamp(self.time * 3600).strftime('%d/%m/%Y, %H:%M')
        else:
            dt = self.time
        return "{}. {} - {}: {}".format(self.i, dt, self.sender, self.content)

class WhatsappParser:
    def __init__(self, processes=None):
        if processes is None:
            self.processes = os.cpu_count()
        else:
            self.processes = processes
        self.messages = []
        self.people = {} # Name to id
        self.time_delta = 0

        self.manager = multiprocessing.Manager()
    
    def __parse_messages(self, lines, return_dict, return_id, start_id=0, chat_id=0):
        messages = []

        for i, l in enumerate(lines):
            l = l.lower()

            if all(_ not in l for _ in omitted_strings):
                try:
                    day, month, year = l.split("/", 3)
                    year, hour = year.split(", ", 1)
                    hour, minute = hour.split(":", 1)
                    minute, sender = minute.split(" - ", 1)
                    sender, content = sender.split(": ", 1)
                    content = content[:-1]

                    if minute[-1] == "m":
                        if minute[-2] == "p":
                            hour = int(hour) + 12

                        minute = minute[:2]

                    if int(year) < 2000:
                        year = int(year) + 2000

                    time = int(datetime.datetime(int(year), int(month), int(day), int(hour), int(minute)).timestamp() / 3600)

                    m = Message(time, sender, content, i=i + start_id, chat_id=chat_id)

                    messages.append(m)
                except ValueError:
                    pass

        return_dict[return_id] = messages
    
    def parse_file(self, filename, chat_id=0):
        '''Parse messages from a Whatsapp chat log using multiprocessing'''

        lines = open(filename, encoding="utf8").readlines()[-2<<14:]

        # Multiprocessing
        chunk_size = len(lines) // self.processes

        return_dict = self.manager.dict()

        jobs = []
        for j in range(self.processes):
            l = lines[j * chunk_size:(j + 1) * chunk_size]
            jobs.append(multiprocessing.Process(target=self.__parse_messages, args=(l, return_dict, j, j * chunk_size, chat_id)))

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        # Flatten the return_dict
        for _ in return_dict.values():
            self.messages = np.concatenate((self.messages, _))

    def encode_senders(self):
        '''Encode the sender names as numbers'''

        # Make a dictionary of all of the people
        c = Counter(map(lambda m: m.sender, self.messages)).most_common()
        c = [_[0] for _ in c]
        for i, p in enumerate(c):
            self.people[p] = i

        for i, m in enumerate(self.messages):
            self.messages[i].sender = self.people[m.sender] 


    def flatten(self):
        '''Concatenate the contents of all of the messages into a single string'''

        return ''.join([_.content for _ in self.messages])

    def normalize_time(self, delta=None):
        '''Normalize the time and date on all the messages'''
        if delta is None:
            delta = - sum([_.time for _ in self.messages]) // len(self.messages)

        self.time_delta += delta
        
        for i, m in enumerate(self.messages):
            self.messages[i].time += delta

    def gen_tokenizer(self, min_frequency=6, limit_alphabet=150):
        '''Create a WordPiece tokenizer from the parsed data'''

        # Store the flattened text in a temporary file
        f = tempfile.NamedTemporaryFile()
        text = self.flatten()
        f.write(text.encode("utf8"))

        # Create the tokenizer
        tokenizer = BertWordPieceTokenizer()
        tokenizer.train([f.name], min_frequency=min_frequency, limit_alphabet=limit_alphabet)
        f.close()

        return tokenizer

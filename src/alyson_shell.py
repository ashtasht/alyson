import os
import glob
from cmd import Cmd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Disable unnecesarry TensorFlow logs
import numpy as np
import tensorflow as tf
from tokenizers import Tokenizer

import whatsapp_parser

processes = None
w_parsers, tokenizers = {}, {}

def enable_gpu():
    # Enable GPU memory growth
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) != 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
        return True
    else:
        return False

class AlysonShell(Cmd):
    prompt = " > "
    intro = "You may use this shell to train your own chatbot, you should read REAMDE.md before starting.\nType help to get a list of commands.\n"

    def do_exit(self, inp):
        '''Exit this shell'''

        return True

    def do_gpu(self, inp):
        '''Enable GPU memory growth'''

        if enable_gpu():
            print("Enabled GPU memory growth")
        else:
            print("Cannot enable GPU memory growth, no GPU avilable")

    def do_processes(self, inp):
        '''Set the amount of processes to use when processing the logs ( > processes <amount>)'''

        try:
            processes = abs(int(inp))
        except ValueError:
            print("Cannot parse the number of processes given")

    def do_ls(self, inp):
        '''List all the files in a directory ( > ls <dir>)'''

        try:
            if inp == "":
                inp = "."
            print(', '.join(os.listdir(inp)))
        except FileNotFoundError:
            print("No such file or directory: \"{}\"".format(inp))
        except NotADirectoryError:
            print("Not a directory: \"{}\"".format(inp))

    def do_parse(self, inp):
        '''Parse multiple Whatsapp log files ( > parse <parser_name> file1 file2 dir/*)'''

        # Parse the Whatsapp parser name
        words = inp.split(" ")
        if words[0] == "":
            print("Invalid parser name")
            return
        parser_name = words[0]
        
        # List the files to parse
        if len(words) == 1:
            print("Invalid usage")
            return

        files = np.array([])
        files = np.concatenate((files, *[glob.glob(_) for _ in words[1:]]))

        if len(files) == 0:
            print("Invalid files")
            return

        w_parsers[parser_name] = whatsapp_parser.WhatsappParser(processes) # Create a new Whatsapp parser

        for i, f in enumerate(files):
            print("Parsing {} ({})...".format(f, i))
            try:
                w_parsers[parser_name].parse_file(f, i)
            except IsADirectoryError:
                print("Skipping \"{}\" as it is not a file...".format(f))

    def do_normalizetime(self, inp):
        '''Normalize time ( > normalizetime <parser>)'''

        # Parse the Whatsapp parser name
        parser_name = inp.split(" ")[0] 
        try:
            w_parsers[parser_name].normalize_time()
        except KeyError:
            print("Unknown parser: {}".format(parser_name))

    def do_parsers(self, inp):
        '''Show a list of defined parsers'''

        [print(k) for k, v in w_parsers.items()]

    def do_showmessage(self, inp):
        '''Print a message by its id ( > showmessage <parser> <id>)'''

        words = inp.split(" ")
        if len(words) != 2:
            print("Invalid usage")
            return

        try:
            i = int(words[1])
            print(w_parsers[words[0]].messages[i])
        except KeyError:
            print("Unknown parser: {}".format(words[0]))
        except ValueError:
            print("Invalid id")

    def do_encodesenders(self, inp):
        '''Give an ID number to each sender ( > encodesenders <parser_name>)'''

        parser_name = inp.split(" ")[0]
        if parser_name == "":
            print("Invalid usage")
            return

        w_parsers[parser_name].encode_senders()

    def do_aparse(self, inp):
        '''Parse multiple files and do additional processing ( > parse <parser_name> file1 file2 dir/*)'''

        # Normal parsing
        self.do_parse(inp)

        # Get the parser name
        parser_name = inp.split(" ")[0]

        # Encode senders as numbers
        w_parsers[parser_name].encode_senders()
        print([_[0] for _ in w_parsers[parser_name].people.items()]) # Show all the known people

        # Normalize the time encoding
        self.do_normalizetime(parser_name)
        print("Time delta: {}".format(w_parsers[parser_name].time_delta))

    def do_gentoken(self, inp):
        '''Create a new WordPiece tokenizer from a parser ( > gentoken <tokenizer_name> <parser_name>)'''

        words = inp.split(" ")
        if len(words) != 2:
            print("Invalid usage")
            return
        
        tokenizer_name, parser_name = words[0], words[1]

        try:
            tokenizers[tokenizer_name] = w_parsers[parser_name].gen_tokenizer()
        except KeyError:
            print("Unknown parser: {}".format(parser_name))
            return

    def do_tokenize(self, inp):
        '''Encode a string using a tokenizer ( > tokenize <tokenizer> <string>)'''

        words = inp.split(" ")
        tokenizer_name = words[0]
        string = " ".join(words[1:])
        
        try:
            t = tokenizers[tokenizer_name].encode(string)
        except KeyError:
            print("Unknown tokenizer: {}".format(tokenizer_name))
            return
        print(t.ids)
        print(t.tokens)

    def do_exporttoken(self, inp):
        '''Export a tokenizer to a file ( > exporttoken <tokenizer> <filename>)'''

        words = inp.split(" ")
        if len(words) != 2:
            print("Invalid usage")
            return

        tokenizer_name, filename = words

        try:
            tokenizers[tokenizer_name].save(filename)
        except KeyError:
            print("Unknown tokenizer: {}".format(tokenizer_name))

        print("Exported to {}".format(filename))

    def do_loadtoken(self, inp):
        '''Load a tokenizer from a file ( > loadtoken <tokenizer> <filename>)'''

        words = inp.split(" ")
        if len(words) != 2:
            print("Invalid usage")
            return

        tokenizer_name, filename = words
        if not os.path.isfile(filename):
            print("No such file: {}".format(filename))

        tokenizers[tokenizer_name] = Tokenizer.from_file(filename)

AlysonShell().cmdloop()

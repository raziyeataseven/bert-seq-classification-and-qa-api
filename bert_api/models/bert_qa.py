import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch


class BertQA:
    def __init__(self):
        super(BertQA, self).__init__()
        
        # importing classification model, using simpletransformers lib which installed at https://colab.research.google.com/drive
        #create model with turkish bert
        self.tokenizer = AutoTokenizer.from_pretrained("lserinol/bert-turkish-question-answering")
        self.model = AutoModelForQuestionAnswering.from_pretrained("lserinol/bert-turkish-question-answering")               

    def answer_question(self, question, answer_content):
        '''
        Takes a `question` string and an `answer_text` string (which contains the
        answer), and identifies the words within the `answer_text` that are the
        answer. Prints them out.
        '''
        # ======== Tokenize ========
        # Apply the tokenizer to the input text, treating them as a text-pair.
        input_ids = self.tokenizer.encode(question, answer_content)

        # Report how long the input sequence is.
        print('Query has {:,} tokens.\n'.format(len(input_ids)))

        # ======== Set Segment IDs ========
        # Search the input_ids for the first instance of the `[SEP]` token.
        sep_index = input_ids.index(self.tokenizer.sep_token_id)

        # The number of segment A tokens includes the [SEP] token istelf.
        num_seg_a = sep_index + 1

        # The remainder are segment B.
        num_seg_b = len(input_ids) - num_seg_a

        # Construct the list of 0s and 1s.
        segment_ids = [0]*num_seg_a + [1]*num_seg_b

        # There should be a segment_id for every input token.
        assert len(segment_ids) == len(input_ids)

        # ======== Evaluate ========
        # Run our example through the model.
        outputs = self.model(torch.tensor([input_ids]), # The tokens representing our input text.
                        token_type_ids=torch.tensor([segment_ids]), # The segment IDs to differentiate question from answer_text
                        return_dict=True) 

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # ======== Reconstruct Answer ========
        # Find the tokens with the highest `start` and `end` scores.
        answer_start = torch.argmax(start_scores)
        answer_end = torch.argmax(end_scores)

        # Get the string versions of the input tokens.
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)

        # Start with the first token.
        answer = tokens[answer_start]

        # Select the remaining answer tokens and join them with whitespace.
        for i in range(answer_start + 1, answer_end + 1):
            
            # If it's a subword token, then recombine it with the previous token.
            if tokens[i][0:2] == '##':
                answer += tokens[i][2:]
            
            # Otherwise, add a space then the token.
            else:
                answer += ' ' + tokens[i]
        

        # We'll use the tokens as the x-axis labels. In order to do that, they all need
        # to be unique, so we'll add the token index to the end of each one.
        token_labels = []
        for (i, token) in enumerate(tokens):
            token_labels.append('{:} - {:>2}'.format(token, i))

        self.token_labels = token_labels

        # Pull the scores out of PyTorch Tensors and convert them to 1D numpy arrays.
        s_scores = start_scores.detach().numpy().flatten()
        e_scores = end_scores.detach().numpy().flatten()

        self.s_scores = s_scores
        self.e_scores = e_scores

        self.start_scores_plot()
        self.end_scores_plot()        
        self.scores_plot()
        return answer



    def start_scores_plot(self):
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (32,16)

        # Create a barplot showing the start word score for all of the tokens.
        ax = sns.barplot(x=self.token_labels, y=self.s_scores, ci=None)

        # Turn the xlabels vertical.
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

        # Turn on the vertical grid to help align words to scores.
        ax.grid(True)

        plt.title('Start Word Scores')
        plt.savefig('Start_Word_Scores.png')


    def end_scores_plot(self):
        # Use plot styling from seaborn.
        sns.set(style='darkgrid')
        sns.set(font_scale=1.5)
        plt.rcParams["figure.figsize"] = (32,16)

        # Create a barplot showing the end word score for all of the tokens.
        ax = sns.barplot(x=self.token_labels, y=self.e_scores, ci=None)

        # Turn the xlabels vertical.
        ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="center")

        # Turn on the vertical grid to help align words to scores.
        ax.grid(True)

        plt.title('End Word Scores')
        plt.savefig('End_Word_Scores.png')


    def scores_plot(self):
        # plt.rcParams["figure.figsize"] = (16,8)
        # Store the tokens and scores in a DataFrame. 
        # Each token will have two rows, one for its start score and one for its end
        # score. The "marker" column will differentiate them. A little wacky, I know.
        scores = []
        for (i, token_label) in enumerate(self.token_labels):
            # Add the token's start score as one row.
            scores.append({'token_label': token_label, 
                        'score': self.s_scores[i],
                        'marker': 'start'})
            
            # Add  the token's end score as another row.
            scores.append({'token_label': token_label, 
                        'score': self.e_scores[i],
                        'marker': 'end'})
            
        df_scores = pd.DataFrame(scores)

        # Draw a grouped barplot to show start and end scores for each word.
        # The "hue" parameter is where we tell it which datapoints belong to which
        # of the two series.
        g = sns.catplot(x="token_label", y="score", hue="marker", data=df_scores,
                        kind="bar", height=6, aspect=4)

        # Turn the xlabels vertical.
        g.set_xticklabels(g.ax.get_xticklabels(), rotation=90, ha="center")

        # Turn on the vertical grid to help align words to scores.
        g.ax.grid(True)
        g.figure.savefig("Start_and_End_Scores.png")

# if __name__=='__main__':
#     clf = BertQA(productId=0, question_class=1) 
#     print(clf.answer_question("bağlantı özellikleri neler?"))

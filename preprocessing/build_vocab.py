#  to get all the Unicodes: https://gist.github.com/ngs/2782436
# -*- coding: utf-8 -*-

import re, string
from fractions import Fraction

def extracting_token(data):

    tokens=[]
    problem_idx=[]

    for idx, eqn in enumerate(data):
        open_angle = [idx_open for idx_open, angle in enumerate(eqn) if angle == '<']
        close_angle = [idx_close for idx_close, angle in enumerate(eqn) if angle == '>']

        for i in range(len(open_angle)):
            try:
                token1 = eqn[open_angle[i]:close_angle[i]+1]
                if token1 not in tokens:
                   tokens.append(token1)
                   # print('token1: ', token1)
                if i<len(open_angle)-1:
                   token2 = eqn[close_angle[i]+1:open_angle[i+1]]
                   token2=token2.strip()
                   if token2 not in tokens:
                       tokens.append(token2)
                       # print('token2: ', token2)
            except:
                problem_idx.append(str(idx))
                pass

    return tokens

def check_if_int_float(num):

    num = num.strip().replace(' ', '')
    try:
        float(num)
        return True
    except:
        try:
            int(num)
            return True
        except:
            try:
                float(sum(Fraction(s) for s in num.split()))
                return True
            except:
                return False

def main():

    data = open('mml.txt').readlines()

    # extract the tokens
    tokens = extracting_token(data)

    # check the frequency and build the final vocab based on that
    final_tokens_list = []

    # add few tokens first
    # final_tokens.write('<pad>'+'\n')
    # final_tokens.write('<unk>'+'\n')
    # final_tokens.write('<sos>'+'\n')
    # final_tokens.write('<eos>'+'\n')
    # final_tokens.write('.'+'\n')
    # final_tokens.write('/'+'\n')
    # final_tokens.write(':'+'\n')

    # adding these tokens to list
    for t in ['<pad>', '<unk>', '<sos>','<eos>','<.>','</>', '<:>']:
        final_tokens_list.append(t)

    for intg in range(0, 10):
        # final_tokens.write(str(intg)+'\n')
        final_tokens_list.append(str(intg))

    for letter in string.ascii_lowercase:
        # final_tokens.write(letter+'\n')
        final_tokens_list.append(letter)

    for letter in string.ascii_uppercase:
        # final_tokens.write(letter+'\n')
        final_tokens_list.append(letter)

    for tok in tokens:
        tok = tok
        count = 0
        for d in data:
            if tok in d:
                count += len([i for i in range(len(d)) if d.startswith(tok, i)])
        if count>=10:
            int_float_flag = check_if_int_float(tok)
            if tok not in final_tokens_list and not int_float_flag:
                # final_tokens.write(tok+'\n')
                final_tokens_list.append(tok)

    # writing set of tokens
    final_tokens = open('vocab.txt', 'w')
    for fin_tok in set(final_tokens_list):
        final_tokens.write(fin_tok+'\n')

if __name__ == '__main__':
    main()

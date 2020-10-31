def remove_punctuation_from_word(selected_word, punctuation_list):
    final_word = ""
    for a in selected_word:
        if a not in punctuation_list:
            final_word += a
    return final_word


punctuation = ['!', '"', "'", '#', '(', ')', '*', '-', ',', '.', '/', ':', '[', ']', '|', ';', '?', '،', '...', '$',
               '{',
               '}', '=', '==', '===', '>', '<', '>>', '<<', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹', '۰', '«', '||',
               '""', "''", "&", "'''", '"""', '»', '', '–', "؛", "^"]

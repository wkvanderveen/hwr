from os.path import join


def write_to_file(posterior_word, filepath, filename):
    print("Writing to %s" % filename)
    length = len(posterior_word)
    with open(join(filepath, filename+".txt"), "a+", encoding="utf-8") as f:
        for i in range(length):
            f.write(str(posterior_word[i]))
        f.write("\n")

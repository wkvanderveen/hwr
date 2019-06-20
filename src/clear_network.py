import os
import shutil

folder = '../../data/'
clear_lines = True
for the_file in os.listdir(folder):
    file_path = os.path.join(folder, the_file)
    try:
        if os.path.isfile(file_path):
            if file_path != folder + "dimensions.txt" and clear_lines:
                os.unlink(file_path)

        if os.path.isdir(file_path):
            if not file_path == folder + "original_letters" and \
               not file_path == folder + "weights" and \
               not file_path == folder + "new-lines" and \
               not (file_path == folder + "lines-test" and not clear_lines) and \
               not (file_path == folder + "lines-train" and not clear_lines) and \
               not file_path == folder:
               shutil.rmtree(file_path)
    except Exception as e:
        print(e)

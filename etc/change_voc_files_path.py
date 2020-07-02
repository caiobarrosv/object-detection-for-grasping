import os
from xml.dom import minidom

# Feito por: Daniel Oliveira

# Esse script converte o path do arquivo .xml para o path que você quiser
# Ele mantém a numeração
# configure o caminho em new_path

# coloque o nome da pasta em
# for root, dirs, files in os.walk("bar clamp"):

def change_path(filename):
    new_path = "new\path\here"
    path = "bar clamp/" + filename
    mydoc = minidom.parse(path)
    items = mydoc.getElementsByTagName('path')[0]
    txt = items.firstChild.data

    txt_replace = "D:\Tensorflow2\Teste de treinamento\dataset\\bar clamp"
    txt_new = txt.replace(txt_replace, new_path)
    mydoc.getElementsByTagName("path")[0].childNodes[0].nodeValue = txt_new
    with open(path, "w") as fs:
        fs.write(mydoc.toxml())
        fs.close()

for root, dirs, files in os.walk("bar clamp"):
    for filename in files:
        if not filename.endswith('.xml'):
            continue
        else:
            change_path(filename)
            print(filename)
import os
import shutil


header = "PFock.h"
input = "pfock_def.h"
includes = ["CInt.h", ]

olddef = "__PFOCK_DEF_H__"
newdef = "__PFOCK_H__"

structs = ["PFock", "Ovl", "CoreH"]

fin = open(input, "r");
fout = open(header, "w");

fout.write("#ifndef " + newdef + "\n");
fout.write("#define " + newdef + "\n");
fout.write("\n\n");

for str in includes:
    fout.write("#include <" + str + ">" + "\n")

fout.write("\n\n");

for str in structs:
    fout.write("struct " + str + ";\n");

for line in fin:
    if  (olddef not in line) and ("#include" not in line):
        fout.write(line);
   
fout.write("#endif " + "/* " + newdef + " */");

fout.close();
fin.close();

shutil.move(header, "../include/" + header)

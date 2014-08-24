#!/bin/sh

for file in $1 
do
echo $file
./sort_mol ../data/cc-pvdz.gbs $file
matlab -nodesktop -nosplash -r "reorder('$file', 'map.dat', 'new.dat');exit;"
#cp -f new.dat $file
#rm -f new.dat
#rm -f map.dat
done

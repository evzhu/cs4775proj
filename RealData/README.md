Upon generating the distance matrices as outlined in the corresponding README in TreeBase.zip,
simply create a copy in this directory as a text file. 

For BioNJ or FastNJ:
Run the following line in terminal:
  python "Method File" -f "distance matrix".txt -o "root" -nwk "outputfile.nwk"
For example, 
   python FastNJ.py -f MidJC.txt -o Pleuroziopsis_sp._MDP494 -nwk FastNJMid.nwk 
Corresponds to running FastNJ from a distance matrix file MidJC.txt
rooted at the taxa Pleuroziopsis_sp._MDP494 and the corresponding output is created in FastNJMid.nwk
Note* the nwk output file does not yet have to exist and will be generated

For Weighbor:
Run python NJML.py -f "distance matrix".txt -o "root"
The output is printed in terminal

For NJML:
Run python NJML.py -f "distance matrix".txt
The output is stored in tree.nwk

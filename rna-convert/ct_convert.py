from dot_structure import dot_structure
import xml.etree.cElementTree as ET
import re

def ct2dot(cts, input_title, file_name):
    try:
        if re.match(">.*.ct", input_title[0]):
            title = input_title[0]
            title = title.replace('.ct', '.dot')
        else:
            title = ">seq_name.dot"


        A = []
        B = []
        seq = ''

        idx = 0
        while idx < len(cts):
            line = cts[idx].split()
            if len(line) >= 6 and line[0] == line[5]:
                if line[5] == '1':  # first line
                    A = [int(line[0])]
                    B = [int(line[4])]
                    seq += line[1]
                else:
                    A.append(int(line[0]))
                    B.append(int(line[4]))
                    seq += line[1]
            idx += 1
        if len(A) > 0:
            s = dot_structure(A, B)

        # print (seq)
        # print (''.join(s))

        with open(file_name + "(dot)", 'w') as d:
            new_file = d.write(title + '\n' + seq + '\n' + ''.join(s))
        print("Conversion from (ct) to (dot) completed successfully!")
    except:
        print("Invalid input format")


def ct2rnaml(cts, input_title, file_name):
    try:
        if re.match("#.*.ct", input_title[0]):
            title = input_title[0]
            title = title.replace('.ct', '').replace('>', '')
        else:
            title = "seq_name"


        A = []
        B = []
        seq = ''

        idx = 0
        while idx < len(cts):
            line = cts[idx].split()
            if len(line) >= 6 and line[0] == line[5]:
                if line[5] == '1':  # first line
                    A = [int(line[0])]
                    B = [int(line[4])]
                    seq += line[1]
                else:
                    A.append(int(line[0]))
                    B.append(int(line[4]))
                    seq += line[1]
            idx += 1

        rnaml = ET.Element("rnaml")
        molecule = ET.SubElement(rnaml, "molecule")
        identity = ET.SubElement(molecule, "identity")
        name = ET.SubElement(identity, "name")
        name.text = str(title)
        sequence = ET.SubElement(molecule, "sequence")
        sequence.set("length", str(len(seq)))
        seq_data = ET.SubElement(sequence, "seq-data")
        seq_data.text = seq
        structure = ET.SubElement(molecule, "structure")

        for a, b in zip(A, B):
            if b > a:
                base_pair = ET.SubElement(structure, "base-pair")

                basep5 = ET.SubElement(base_pair, "base-id-p5")
                baseidp5 = ET.SubElement(basep5, "base-id")
                position5 = ET.SubElement(baseidp5, "position")
                position5.text = str(a)

                basep3 = ET.SubElement(base_pair, "base-id-p3")
                baseidp3 = ET.SubElement(basep3, "base-id")
                position3 = ET.SubElement(baseidp3, "position")
                position3.text = str(b)

        tree = ET.ElementTree(rnaml)
        tree.write(file_name + ".xml")
        print("Conversion from (ct) to (rnaml) completed successfully!")
    except:
        print("Invalid input format")


def ct2bpseq(cts, input_title, file_name):
    try:
        if re.match("#.*.ct", input_title[0]):
            title = input_title[0]
            title = title.replace('.ct', '.bpseq')
        else:
            title = ">seq_name.bpseq"

        idx = 0
        string = []
        while idx < len(cts):
            line = cts[idx].split()
            if len(line) >= 6 and line[0] == line[5]:
                string.append(line[0] + ' ' + line[1] + ' ' + line[4])
            idx += 1
        # print('\n'.join(string))


        with open(file_name + '(bpseq)', 'w') as d:
            new_file = d.write(title + '\n' + '\n'.join(string))
        print("Conversion from (ct) to (bpseq) completed successfully!")
    except:
        print("Invalid input format")

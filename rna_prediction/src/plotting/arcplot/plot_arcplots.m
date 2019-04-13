delimiterIn = ' ';
headerlinesIn = 1;
mfe_ct = importdata('/home/marc/PycharmProjects/RNA_Prediction/Data/16Testsequences/data/CT/celegansM-mfe.ct');
native_ct = importdata('/home/marc/PycharmProjects/RNA_Prediction/Data/16Testsequences/data/CT/celegansM-native-nop.ct');
mfe_ct = mfe_ct.data;
native_ct = native_ct.data;
counts = ones(size(mfe_ct(:,1)));

plot_jvizstyle(mfe_ct, native_ct, counts, counts)
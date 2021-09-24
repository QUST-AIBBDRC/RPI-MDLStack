clear all
clc
fid=fopen('Protein_sequence.txt');
string=fscanf(fid,'%s'); %文件输入
%匹配的字符串
firstmatches=findstr(string,'>')+6;%开始位置
endmatches=findstr(string,'>')-1;
firstnum=length(firstmatches); %firstnum=endnum序列的条数
endnum=length(endmatches);

Dim_number=67

  for k=1:Dim_number
    j=1;
    jj=1;
    lensec(k)=endmatches(k+1)-firstmatches(k)+1;%每条序列的长度
    for nnn=1:2
    sign=[ '>',num2str(k)];
    sequ(2*k-1,jj)=sign(nnn);
    jj=jj+1;
    end
   for mm=firstmatches(k):endmatches(k+1)
        sequ(2*k,j)=string(mm); %字符序列
        j=j+1;
   end
  end

   x=cellstr(sequ);
RPI_protein_N_lensec=lensec

for i=1:WEISHU
    nnn=num2str(i);
    name = strcat(nnn,'.pssm');
    fid{i}=importdata(name);
end
C={};
for t=1:WEISHU
    shu=fid{t}.data;
    shuju=shu(1:RPI_protein_N_lensec(1,t),1:20);
    RPI_protein_N_PSSM{t}=shuju;
end

pssm=[];
maxlen=[];
for i=1:numel(RPI_protein_N_PSSM)
    data=RPI_protein_N_PSSM{i};
    data= 1.0 ./ ( 1.0 + exp(-data) );
    pssm=[pssm;data];  %%%%把所有的PSSM文件前L行20列保存到一个文件里
    [row,column]=size(data);
    maxlen=[maxlen;row];
    data=[];
    row=[];
end
index_PA=cumsum(maxlen);   %%%cumsum函数通常用于计算一个数组各行的累加值，index_PA得到所有的行数
maxlen=[];
index_PA=[0;index_PA];
[m,n]=size(index_PA);
save tpc1_python_PSSM.mat pssm index_PA 






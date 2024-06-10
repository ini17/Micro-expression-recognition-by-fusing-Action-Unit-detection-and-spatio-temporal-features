% Read SMIC dataset template

foldername = 'D:\Database\SMIC\SMIC_all_cropped\HS\';
f = dir(foldername);
fsize = size(f);
stride = 2;
for i = 3:fsize(1)
    subname = [foldername, f(i).name,'\micro\'];
    sub = dir(subname);
    for j = 3:5
        typename = [subname, sub(j).name,'\'];
        type = dir(typename);
        typesize = size(type);
        for k = 3:typesize(1)
            filename = [typename, type(k).name,'\']
            file = dir(filename);
            vdata = ReadVoldata_smic(filename);
            of = computOpticalFlow(vdata, stride);
            save(['SMIC_stride2\\of_',type(k).name], 'of');
        end
    end
end
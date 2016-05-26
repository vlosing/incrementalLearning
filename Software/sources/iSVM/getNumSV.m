function numSV = getNumSV()
global models;
MARGIN    = 1;
ERROR     = 2;

numSV = 0;
for classIdx=1:size(models, 2)
    if ~models{classIdx}.isEmpty
        numSV = numSV + length(models{classIdx}.ind{MARGIN}) + length(models{classIdx}.ind{ERROR});
    end
end


function result = allModelsEmpty()
global models;
result = true;
for i=1:size(models, 2)
    if ~models{i}.isEmpty
        result = false;
    end
end
end
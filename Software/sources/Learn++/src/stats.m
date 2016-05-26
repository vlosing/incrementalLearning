function [f_measure,g_mean,recall,precision,err] = stats(f, h, mclass)
%   [f_measure,g_mean,recall,precision,err] = stats(f, h, mclass)
%     @f - vector of true labels
%     @h - vector of predictions on f
%     @mclass - number of classes
%     @f_measure
%     @g_mean
%     @recall
%     @precision
%     @err
%

%     stats.m
%     Copyright (C) 2013 Gregory Ditzler
% 
%     This program is free software: you can redistribute it and/or modify
%     it under the terms of the GNU General Public License as published by
%     the Free Software Foundation, either version 3 of the License, or
%     (at your option) any later version.
% 
%     This program is distributed in the hope that it will be useful,
%     but WITHOUT ANY WARRANTY; without even the implied warranty of
%     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
%     GNU General Public License for more details.
% 
%     You should have received a copy of the GNU General Public License
%     along with this program.  If not, see <http://www.gnu.org/licenses/>.

F = index2vector(f, mclass);
H = index2vector(h, mclass);

recall = compute_recall(F, H, mclass);
err = 1 - sum(diag(H'*F))/sum(sum(H'*F));
precision = compute_precision(F, H, mclass);
g_mean = compute_g_mean(recall, mclass);
f_measure = compute_f_measure(F, H, mclass);

function g_mean = compute_g_mean(recall, mclass)
g_mean = (prod(recall))^(1/mclass);

function f_measure = compute_f_measure(F, H, mclass)
f_measure = zeros(1, mclass);
for c = 1:mclass
  f_measure(c) = 2*F(:, c)'*H(:, c)/(sum(H(:, c)) + sum(F(:, c))); 
end
f_measure(isnan(f_measure)) = 1;

function precision = compute_precision(F, H, mclass)
precision = zeros(1, mclass);
for c = 1:mclass
  precision(c) = F(:, c)'*H(:, c)/sum(H(:, c)); 
end
precision(isnan(precision)) = 1;

function recall = compute_recall(F, H, mclass)
recall = zeros(1, mclass);
for c = 1:mclass
  recall(c) = F(:, c)'*H(:, c)/sum(F(:, c)); 
end
recall(isnan(recall)) = 1;

function y = index2vector(x, mclass)
y = zeros(numel(x), mclass);
for n = 1:numel(x)
  y(n, x(n)) = 1;
end

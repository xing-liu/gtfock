function reorder(input_filename, map_filename, output_filename)

fid = fopen(input_filename, 'r');
line1 = fgets(fid); % get the first line
np = sscanf(line1, '%d');
line2 = fgets(fid); % get the comment line
fprintf('number of particles: %d\n', np);

lines = cell(np);
pos = zeros(3,np);

% read positions
for i=1:np
  lines{i} = fgets(fid);
  pos(:,i) = sscanf(lines{i}, ' %*c %f %f %f'); % suppress atom name
end
fclose(fid);

% compute the ordering
map = load(map_filename);
sp_map = spconvert (map);
r = symrcm(sp_map);
spy(sp_map(r, r));


% write output file
fid = fopen(output_filename, 'w');
fprintf(fid, '%s', line1);
fprintf(fid, '%s', line2);
for i=1:np
  fprintf(fid, '%s', lines{r(i)}); % print in new order
end
fclose(fid);

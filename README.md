# Data-Computer-Science

select genre, count(*) as movie_id from movies_genres group by genre order by movie_id;
select concat(first_name,' ',last_name) as 'Name' from (select * from actors join roles on actors.id = roles.actor_id) as new_table_name where role = 'Wonder Woman';
select * from movies where rank >= 9.5 and year >= 2007;
select first_name, last_name, name from ((select * from directors join movies_directors on directors.id = movies_directors.director_id) as new_table join movies on new_table.movie_id = movies.id) where name like 'Zz';
select first_name, last_name, name from ((select * from directors join movies_directors on directors.id = movies_directors.director_id) as new_table join movies on new_table.movie_id = movies.id) where rank >= 9.5;
select concat(first_name,' ',last_name) as 'Name' from ((select * from actors join roles on actors.id = roles.actor_id where role = concat(first_name,' ',last_name))as new_table);
select first_name, last_name, avg(rank) from ((select * from directors join movies_directors on directors.id = movies_directors.director_id) as new_table join movies on new_table.movie_id = movies.id) group by first_name,last_name;
select  first_name,last_name,max(rank) from ((select * from directors join movies_directors on directors.id = movies_directors.director_id) as new_table  join movies on new_table.movie_id = movies.id) group  by first_name,last_name order by max(rank) desc limit 1 ;
create view max_role as select first_name,last_name from ((select first_name,last_name,id, count(*) as m_role from ((select* from actors join roles on actors.id = roles.actor_id) as new_table) group by first_name, last_name,id order by m_role desc)as new_newest_table) where m_role>300;
select * from max_role limit 1;
select name,count(name) from (select * from movies join movies_genres on movies.id = movies_genres.movie_id where genre = 'Comedy' union all select * from movies join movies_genres on movies.id = movies_genres.movie_id where genre = 'Action') as newest_table group by name having count(name)>1 ;
select name,count(name) from (select * from movies join movies_genres on movies.id = movies_genres.movie_id where genre in ('Comedy') union all select * from movies join movies_genres on movies.id = movies_genres.movie_id where genre in ('Action')) as newest_table group by name having count(name)>1 ;
select directors.first_name,directors.last_name,movies.name,roles.role from directors join movies_directors on directors.id = movies_directors.director_id join movies on movies_directors.movie_id = movies.id join actors on directors.first_name = actors.first_name and directors.last_name = actors.last_name join roles on actors.id = roles.actor_id;
select genre, count(*) as movie_id from movies_genres group by genre order by movie_id desc limit 1;
select first_name,last_name from(select* from actors join (select actor_id,count(distinct role),count(distinct movie_id) from (select * from roles union all select * from roles where roles.actor_id = roles.actor_id and roles.movie_id = roles.movie_id and (roles.role != roles.role))as newerest group by actor_id having  count(distinct role)>1 and count(distinct movie_id) = 1) as newnewnew on actors.id = newnewnew.actor_id) as newernew123;
select * from movies order by rank desc limit 25;

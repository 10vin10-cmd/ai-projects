/*With RECURSIVE Rootquery as (
select '1' as root, routeid, origin, destination, ticketcost from BusRoutes 

union 
SELECT root + ',' , ch.routeid, ch.origin, ch.destination, ch.ticketcost
FROM Rootquery R join BusRoutes ch on R.destination = ch.origin and R.routeid != ch.routeid
)
select * from Rootquery R join BusRoutes b on R.destination = b.destination
order by R.root, R.ticketcost
*/

WITH RECURSIVE Rootquery AS (
    -- Anchor Member: Initialize path tracking and cost accumulation
    SELECT 
        routeid::text AS path,
        ARRAY[routeid] AS visited_routes, -- Array to keep track of all visited route IDs
        routeid, 
        origin, 
        destination, 
        ticketcost AS total_ticketcost
    FROM BusRoutes   
    
    UNION ALL
    
    -- Recursive Member: Chain routes together while avoiding cyclic loops
    SELECT 
        R.path || ',' || ch.routeid::text AS path, 
        R.visited_routes || ch.routeid AS visited_routes, -- Append new route ID to array
        ch.routeid, 
        ch.origin, 
        ch.destination, 
        R.total_ticketcost + ch.ticketcost AS total_ticketcost
    FROM Rootquery R 
    JOIN BusRoutes ch ON R.destination = ch.origin 
    -- Cycle prevention: Stop if the next routeid is ALREADY in our visited array
    WHERE NOT (ch.routeid = ANY(R.visited_routes))
)
SELECT 
    R.path,
    R.origin AS path_start,
    R.destination AS path_end,
    R.total_ticketcost,
    b.routeid AS matching_dest_routeid,
    b.origin AS matching_dest_origin
FROM Rootquery R 
JOIN BusRoutes b ON R.destination = b.destination 
ORDER BY R.path, R.total_ticketcost;


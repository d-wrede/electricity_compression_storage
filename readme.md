# Generic storage setup

                      b_el						  b_PV
              +----------------+				  +----+
demand  <--   |                |  <-- PV_link <-- |    |  <-- PV
              |                |				  |    |
              |                |				  |    |  --> excess_sink
              |                |				  +----+
			  |                |
battery <-->  |                |  <-- grid_source
              +----------------+


# Compression storage setup

                      b_el						     b_PV
				+----------------+				    +----+
demand  <-- 	|                |  <-- PV_link <-- |    |  <-- PV
				|                |				    |    |
				|                |					|    |  --> feed_in / excess sink
				|                |					+----+
				|                |				  	    			  b_air
				|                |									 +----+
grid_source --> |                |  -->   compression_converter -->  |    | <--> storage
				|                |							 |		 |    |
				|                |  <-- expansion_converter <|-----  |    |
				+----------------+					|		 |		 +----+
													|        v
												    |     b_heat --> heat_sink
													v
												 b_cold --> cold_sink
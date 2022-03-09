This folder contains three scripts, five csv data files, and one HTML visualization,
    - Scripts
        + main.py
            * Script for clustering and visualization generation
        + merge_edges.py
            * Script for merge previous people co-mention edge list with 
                new organization edge list.
        + join_cluster_assignments.py
            * Script for join cluster assignments to edge list
    - Data
        + rw_tone_org.csv
            * New edge list by Zhaosen incorporating the companies. However, 
                since the articles used for calculating these scores 
                must contain at least one interested companies, the articles
                missed a lot of people-only co-mentions.
        + rw_tone_ppl.csv
            * Previously calculated people-only co-mentions. I merge this
                with `rw_tone_org.csv` to create `rw_tone_merge.csv`
        + rw_tone_merge.csv
            * Concatenated/merged edge list
        + rw_tone_cls_assignment.csv
            * A table with two columns: entity_name, cluster (ids)
        + rw_tone_merge_cls_assignment.csv
            * Join cluster assignments with edge list

    - Visualization
        + rw_tone_cls.html
            * Useful visualization configs:
            * zoom:
                = place your mouse on the graph and use your scroll to zoom in and out
            * nodes:
                = change font <size> and <strokeColor> to make node labels clearer
            * edges:
                = check <smooth> and select from <type> drop-down the <dynamic> type. I
                    personally find this plot looking better
            * physics:
                = uncheck <enabled> to stop physics rendering so you can drag the nodes
                    around and pin them down
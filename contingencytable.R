UFo <- read.csv("C:/Users/David/Documents/GitHub/Blue2_HW6_UFO_Text/ufo_plot_data.csv")

UFo['label'] = Ufo['label'] + 1 

# 2-Way Cross Tabulation
library(gmodels)
CrossTable(UFo$label, UFo$state, chisq=TRUE, expected = TRUE)

UFo$label = UFo$label + 1 

install.packages("sjPlot")
library(sjPlot)


sjt.xtab(UFo$label, UFo$state, title = "Contingency Table of Cluster vs State",
         var.labels=c("Cluster #", "State"),
         show.exp=TRUE, show.summary=TRUE,statistics="cramer",show.legend= TRUE, show.cell.prc = TRUE)

library(factoextra)
library(NbClust)
# Elbow method
fviz_nbclust(df, kmeans, method = "wss") +
  geom_vline(xintercept = 4, linetype = 2)+
  labs(subtitle = "Elbow method")

# Silhouette method
fviz_nbclust(df, kmeans, method = "silhouette")+
  labs(subtitle = "Silhouette method")

# Gap statistic
# nboot = 50 to keep the function speedy. 
# recommended value: nboot= 500 for your analysis.
# Use verbose = FALSE to hide computing progression.
set.seed(123)
fviz_nbclust(df, kmeans, nstart = 25,  method = "gap_stat", nboot = 50)+
  labs(subtitle = "Gap statistic method")

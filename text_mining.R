````{r setup, include=FALSE}
library(reticulate)
os <- import("os")
os$listdir(".")

```{python}

```

```{R}

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

{Python}
dasd 
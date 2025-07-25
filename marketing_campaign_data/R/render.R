library(rmarkdown)

render("01_eda.Rmd")
render("02_stat_tests.Rmd")
render("03_modeling.Rmd")
render("04_profiling.Rmd")

render("master.Rmd")
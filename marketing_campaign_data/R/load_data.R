
convert_bank_data_factors <- function(data) {
    data$y <- factor(data$y, levels = c("no", "yes"))
    data$job <- factor(data$job)
    data$marital <- factor(data$marital)
    data$education <- factor(
        data$education,
        levels = c("unknown", "primary", "secondary", "tertiary"),
        ordered = TRUE
    )
    data$default <- factor(data$default, levels = c("no", "yes"))
    data$housing <- factor(data$housing, levels = c("no", "yes"))
    data$loan <- factor(data$loan, levels = c("no", "yes"))
    data$contact <- factor(data$contact)
    data$month <- factor(
        data$month,
        levels = c("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        ordered = TRUE
    )
    data$poutcome <- factor(data$poutcome)
    return(data)
}

load_bank_data <- function(filename = "bank.csv") {
    if (!file.exists(here::here("data", filename))) {
        stop("bank.csv file not found in /data directory.")
    }
    data <- read.csv(here::here("data", filename), sep = ";")
    
    return(
        convert_bank_data_factors(data)
    )
}
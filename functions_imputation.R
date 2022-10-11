# Helper Functions for Imputation Rmd

check.dim <- function(data) {
  if (is.na(dim(data)[1])) {
    data <- matrix(data, length(data), 1)
  }
  return(data)
}


#' Simple Imputation: Mean
#'
#' @param df data frame with missing values 
#'
#' @return data frame with missingness filled with mean 
#' 
imp_mean <- function(df){
  df <- check.dim(df)
  
  for (i in 1:dim(df)[2]){
    loc_na <- which(is.na(df[ , i]))
    df[loc_na, i] <- mean(df[ , i], na.rm = T)
  }
  return(df)
}



#' Add missingness to data frame
#' 
#' @param df       data frame object  
#' @param prop     proportion of missingness
#' @param misstype missingness type: mcar or mnar
#'
#' @return         data frame with missingness added
#' 
add_miss <- function(df, prop, misstype = "mcar"){
  df <- check.dim(df)
  
  for (i in 1:dim(df)[2]){
    df[,i] <- add_miss_v(v_x      = df[,i],
                         prop     = prop,
                         misstype = misstype)
  }
  return(df)
}



#' Add Missingness to a vector
#'
#' @param v_x      vector to add missingness to 
#' @param prop     proportion missing
#' @param misstype type of missingness: mcar or mnar
#'
#' @return         vector with missingness added
#' 
add_miss_v <- function(v_x, prop, misstype = "mcar"){
  n_miss <- rbinom(1, length(v_x), prop)
  if (misstype == "mcar"){
    miss_indx <- sample(1:length(v_x), n_miss, replace = F)
  } else if (misstype == "mnar"){
    # Missing not at random - remove higher values
    miss_indx <- order(v_x, decreasing = T)[1:n_miss] 
  }
  v_x[miss_indx] <- NA
  return(v_x)
}



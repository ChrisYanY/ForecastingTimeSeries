get.quotes <- function(ticker,
 from=(Sys.Date()- 365 * 4),
 to=(Sys.Date()),
 interval="d") {
 # define parts of the URL
 base <- "http://ichart.finance.yahoo.com/table.csv?";
 symbol <- paste("s=", ticker, sep="");
 # months are numbered from 00 to 11, so format the month correctly
 from.month <- paste("&a=",
 formatC(as.integer(format(from,"%m"))-1,width=2,flag="0"),
 sep="");
 from.day <- paste("&b=", format(from,"%d"), sep="");
 from.year <- paste("&c=", format(from,"%Y"), sep="");
 to.month <- paste("&a=",
 formatC(as.integer(format(to,"%m"))-1,width=2,flag="0"),
 sep="");
 to.day <- paste("&e=", format(to,"%d"), sep="");
 to.year <- paste("&f=", format(to,"%Y"), sep="");
 inter <- paste("&g=", interval, sep="");
 last <- "&ignore=.csv";
 # put together the URL
 url <- paste(base, symbol, from.month, from.day, from.year,
 to.month, to.day, to.year, inter, last, sep="");
 # get the file
 message("url is:", url)
 tmp <- read.csv(url);
 # add a new column with ticker symbol labels
 cbind(symbol=ticker,tmp);

}


get.multiple.quotes <- function(tkrs,
 from=(Sys.Date()- 365 * 4),
 to=(Sys.Date()),
 interval="d") {
 tmp <- NULL;
 for (tkr in tkrs) {
 if (is.null(tmp))
 tmp <- get.quotes(tkr,from,to,interval)
 else tmp <- rbind(tmp,get.quotes(tkr,from,to,interval))
 }
 tmp
}

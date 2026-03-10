cat("Hello from R runtime\n")

timestamp <- Sys.time()

df <- data.frame(
  message = "Hello from R runtime",
  timestamp = timestamp
)

output_path <- "R/outputs/hello_r_output.csv"

write.csv(df, output_path, row.names = FALSE)

cat("Output file written to:", output_path, "\n")
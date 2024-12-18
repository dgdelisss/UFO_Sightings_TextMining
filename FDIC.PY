library(ggplot2)
library(gganimate)
library(dplyr)
library(tidyr)

# Sample transaction data
# Columns: from_account, to_account, amount, month
transactions_data <- data.frame(
  from_account = c("Account A", "Account B", "Account C", "Account A", "Account B"),
  to_account = c("Account B", "Account C", "Account A", "Account C", "Account A"),
  amount = c(1000, 1500, 750, 2000, 1250),
  month = c(1, 1, 2, 2, 3)
)

# Calculate account balances over time
calculate_balances <- function(transactions) {
  # Initialize starting balances
  start_balances <- data.frame(
    account = unique(c(transactions$from_account, transactions$to_account)),
    balance = 10000  # Starting balance for each account
  )
  
  # Track balances over months
  monthly_balances <- lapply(1:max(transactions$month), function(m) {
    month_transactions <- transactions[transactions$month == m, ]
    
    # Subtract amounts sent
    sent_amounts <- month_transactions %>%
      group_by(from_account) %>%
      summarise(total_sent = sum(amount))
    
    # Add amounts received
    received_amounts <- month_transactions %>%
      group_by(to_account) %>%
      summarise(total_received = sum(amount))
    
    # Update balances
    current_balances <- start_balances %>%
      left_join(sent_amounts, by = c("account" = "from_account")) %>%
      left_join(received_amounts, by = c("account" = "to_account")) %>%
      mutate(
        total_sent = replace_na(total_sent, 0),
        total_received = replace_na(total_received, 0),
        balance = balance - total_sent + total_received
      ) %>%
      select(account, balance)
    
    # Update start balances for next iteration
    start_balances <<- current_balances
    
    # Add month column
    current_balances$month <- m
    return(current_balances)
  })
  
  # Combine monthly balances
  do.call(rbind, monthly_balances)
}

# Generate monthly account balances
account_balances <- calculate_balances(transactions_data)

# Prepare data for visualization
viz_data <- account_balances %>%
  mutate(
    x = case_when(
      account == "Account A" ~ 0.2,
      account == "Account B" ~ 0.5,
      account == "Account C" ~ 0.8
    ),
    y = 0.5,  # Fixed y-coordinate for simplicity
    month = as.factor(month)
  )

# Create the animation
transaction_animation <- ggplot(viz_data, aes(x = x, y = y, size = balance, color = account)) +
  geom_point(alpha = 0.7) +
  scale_size_continuous(range = c(5, 20)) +  # Adjust circle size range
  labs(
    title = "Account Balances Over Time",
    subtitle = "Month: {current_frame}",
    x = "Account Position",
    y = "",
    size = "Balance",
    color = "Account"
  ) +
  theme_minimal() +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank()
  ) +
  # Animation specific layers
  transition_manual(month) +
  enter_fade() +
  exit_fade()

# Save the animation
animate(transaction_animation, 
        nframes = length(unique(viz_data$month)), 
        fps = 1,  # 1 frame per second
        width = 800, 
        height = 400)

# Optional: Save as a gif
# gganimate::anim_save("account_transactions.gif")

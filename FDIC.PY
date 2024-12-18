# Required Libraries
library(ggplot2)
library(gganimate)
library(dplyr)
library(tidyr)
library(transformr)  # For smooth transitions
library(gifski)      # For gif rendering

# Transaction Data Structure
transaction_data <- list(
  source_nodes = data.frame(
    id = c("Source1", "Source2"),
    name = c("Business Income", "Investment Returns"),
    initial_balance = c(50000, 75000),
    stringsAsFactors = FALSE
  ),
  destination_nodes = data.frame(
    id = c("Dest1", "Dest2", "Dest3"),
    name = c("Operating Expenses", "Employee Salaries", "Research & Development"),
    initial_balance = c(30000, 45000, 25000),
    stringsAsFactors = FALSE
  ),
  transactions = data.frame(
    month = c(1, 1, 1, 2, 2, 2, 3, 3, 3),
    from = c("Source1", "Source1", "Source2", 
             "Source1", "Source2", "Source2", 
             "Source1", "Source2", "Source1"),
    to = c("Dest1", "Dest2", "Dest3", 
           "Dest2", "Dest1", "Dest3", 
           "Dest1", "Dest2", "Dest3"),
    amount = c(15000, 20000, 25000, 
               18000, 12000, 30000, 
               22000, 15000, 10000),
    stringsAsFactors = FALSE
  )
)

# Prepare Node Positions
prepare_node_positions <- function(source_nodes, destination_nodes) {
  # Combine source and destination nodes
  all_nodes <- rbind(
    transform(source_nodes, 
              x = 0.2, 
              y = seq(0.2, 0.8, length.out = nrow(source_nodes)),
              group = 'source'),
    transform(destination_nodes, 
              x = 0.8, 
              y = seq(0.2, 0.8, length.out = nrow(destination_nodes)),
              group = 'destination')
  )
  return(all_nodes)
}

# Calculate Monthly Balances
calculate_monthly_balances <- function(nodes, transactions, month) {
  # Reset to initial balances
  current_balances <- nodes
  
  # Filter transactions up to and including current month
  month_transactions <- transactions[transactions$month <= month, ]
  
  # Update balances
  for (i in 1:nrow(month_transactions)) {
    from_idx <- which(current_balances$id == month_transactions$from[i])
    to_idx <- which(current_balances$id == month_transactions$to[i])
    
    current_balances$initial_balance[from_idx] <- 
      current_balances$initial_balance[from_idx] - month_transactions$amount[i]
    current_balances$initial_balance[to_idx] <- 
      current_balances$initial_balance[to_idx] + month_transactions$amount[i]
  }
  
  return(current_balances)
}

# Track Historical Connections
track_historical_connections <- function(transactions) {
  unique_connections <- unique(
    data.frame(
      from = transactions$from, 
      to = transactions$to
    )
  )
  return(unique_connections)
}

# Create Visualization
create_transaction_network <- function(transaction_data) {
  # Prepare all nodes
  all_nodes <- prepare_node_positions(
    transaction_data$source_nodes, 
    transaction_data$destination_nodes
  )
  
  # Track historical connections
  historical_connections <- track_historical_connections(
    transaction_data$transactions
  )
  
  # Prepare data for animation
  animation_frames <- lapply(1:3, function(month) {
    # Calculate monthly balances
    monthly_balances <- calculate_monthly_balances(
      all_nodes, 
      transaction_data$transactions, 
      month
    )
    
    # Current month's transactions
    current_transactions <- transaction_data$transactions[
      transaction_data$transactions$month == month, 
    ]
    
    # Combine data
    data.frame(
      month = month,
      rbind(
        monthly_balances,
        current_transactions
      )
    )
  })
  
  # Combine all frames
  combined_data <- do.call(rbind, animation_frames)
  
  # Color scales
  source_colors <- c("#1f77b4", "#ff7f0e")
  dest_colors <- c("#2ca02c", "#d62728", "#9467bd")
  
  # Balance scaling function
  scale_balance <- function(balance, min_val, max_val) {
    scales::rescale(balance, to = c(20, 60), from = c(min_val, max_val))
  }
  
  # Create the plot
  p <- ggplot() +
    # Historical connections (grey lines)
    geom_segment(
      data = historical_connections,
      aes(
        x = all_nodes$x[match(from, all_nodes$id)], 
        y = all_nodes$y[match(from, all_nodes$id)],
        xend = all_nodes$x[match(to, all_nodes$id)],
        yend = all_nodes$y[match(to, all_nodes$id)]
      ),
      color = "lightgrey",
      alpha = 0.5,
      linewidth = 1
    ) +
    # Active transaction lines
    geom_segment(
      data = combined_data[combined_data$month == month, ],
      aes(
        x = all_nodes$x[match(from, all_nodes$id)], 
        y = all_nodes$y[match(from, all_nodes$id)],
        xend = all_nodes$x[match(to, all_nodes$id)],
        yend = all_nodes$y[match(to, all_nodes$id)]
      ),
      color = "darkgrey",
      alpha = 0.7,
      linewidth = 2
    ) +
    # Nodes
    geom_point(
      data = all_nodes,
      aes(
        x = x, 
        y = y, 
        size = scale_balance(
          initial_balance, 
          min(all_nodes$initial_balance), 
          max(all_nodes$initial_balance)
        ),
        color = id
      ),
      alpha = 0.7
    ) +
    # Node labels
    geom_text(
      data = all_nodes,
      aes(x = x, y = y, label = name),
      color = "white",
      size = 3
    ) +
    # Color scales
    scale_color_manual(
      values = c(
        source_colors[1:length(transaction_data$source_nodes$id)],
        dest_colors[1:length(transaction_data$destination_nodes$id)]
      )
    ) +
    # Theming
    theme_minimal() +
    theme(
      legend.position = "none",
      axis.title = element_blank(),
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank()
    ) +
    # Coordinate system
    coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
    # Animation
    transition_manual(month) +
    labs(title = "Transaction Network - Month: {current_frame}")
  
  # Render animation
  animate(
    p, 
    nframes = 3, 
    fps = 1,
    width = 800, 
    height = 600,
    renderer = gifski_renderer()
  )
}

# Generate the animation
transaction_animation <- create_transaction_network(transaction_data)

# Save the animation
anim_save("transaction_network_animation.gif", transaction_animation)

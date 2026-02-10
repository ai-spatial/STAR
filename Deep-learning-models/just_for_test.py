def scan(y_true_value, true_pred_value, min_sample,
         connected = True, g_grid = None, X_dim = None, step_size = None,
         flex = FLEX_OPTION):
  c,b = get_c_b(y_true_value, true_pred_value)

  max_iteration = 1000

  #init q
  q = np.zeros(y_true_value.shape[1])
  q_init = np.nan_to_num(c/b)
  for i in range(q_init.shape[1]):
    q_class = q_init[:,i]
    s_class, _ = get_top_cells(q_class)
    q[i] = np.sum(c[s_class,i]) / np.sum(b[s_class,i])

  # q = np.random.rand(y_true_value.shape[1])*2
  # q = np.exp(q)

  q_filter = np.sum(b,0) < min_sample
  q[q_filter == 1] = 1
  q[q == 0] = 1
  q = np.expand_dims(q,0)

  log_lr_prev = 0
  for i in range(max_iteration):#coordinate descent
    #update location
    g = c * np.log(q) + b * (1-q)
    g = np.sum(g, 1)
    s0, s1 = get_top_cells(g)
    log_lr = np.sum(g[s0])

    #update q
    q = np.nan_to_num(np.sum(c[s0],0) / np.sum(b[s0],0))
    q[q == 0] = 1
    q[q_filter == 1] = 1

    if log_lr < 0:
      print("log_lr < 0: check initialization!")

    # if (log_lr - log_lr_prev) / log_lr_prev < 0.05:
    #   break

    log_lr_prev = log_lr

    s0 = s0.reshape(-1)
    s1 = s1.reshape(-1)
    
    if (i == max_iteration - 1) and (connected == True):
      s0, s1 = get_connected_top_cells(g, g_grid, X_dim, step_size, flex = flex)

  return s0, s1

def scan_regression(loss_value, min_sample,
        connected = True, g_grid = None, X_dim = None, step_size = None,
        flex = FLEX_OPTION):#y_true_value, true_pred_value
  
  # Different from scan(): expects a precomputed loss vector directly.
  # confirm if input is loss value or negated loss

  g = loss_value

  # Different from scan(): single-pass sort/split instead of coordinate descent.
  sorted_g = np.argsort(g,0)#second input might not be needed
  sorted_g = sorted_g[::-1]
  set_size = np.ceil(sorted_g.shape[0]/2).astype(int)

  sorted_g_value = np.sort(g)#second input might not be needed
  sorted_g_vaule = sorted_g_value[::-1]

  # Different from scan(): flex adjusts split size via variance minimization.
  if flex:
    flex_ratio = FLEX_RATIO
    min_size = (np.ceil(set_size * (1 - flex_ratio))).astype(int)
    max_size = (np.ceil(set_size * (1 + flex_ratio))).astype(int)

    #data for this step will be processed/grided and small; if needed, implement incremental updates
    optimal_size = set_size
    min_variance = np.var(sorted_g_value[0:set_size]) + np.var(sorted_g_value[set_size:])
    for size in range(min_size, max_size):
      variance = np.var(sorted_g_value[0:size]) + np.var(sorted_g_value[size:])
      if variance < min_variance:
        optimal_size = size
        min_variance = variance

    set_size = optimal_size

  # Different from scan(): no q update, just top/bottom index sets.
  s0 = sorted_g[0:set_size].astype(int)
  s1 = sorted_g[set_size:].astype(int)

  s0 = s0.reshape(-1)
  s1 = s1.reshape(-1)

  return s0, s1

  if (i == max_iteration - 1) and (connected == True):
    s0, s1 = get_connected_top_cells(g, g_grid, X_dim, step_size, flex = flex)

  return s0, s1
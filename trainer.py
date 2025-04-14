import torch
import torch.nn.functional as F

def train_model(model, train_loader, val_loader, args):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_loss = float('inf')
    model.to(args.device)

    for epoch in range(args.num_epochs):
        model.train()
        train_loss, train_com, train_corr = 0, 0, 0
        for batch in train_loader:
            if args.task == 'fitb':
                imgs_list, input_ids_list, attention_mask_list, lengths, answers = batch
                batch_size = len(lengths)
                loss = 0
                scores = []
                for i in range(4):
                    imgs = imgs_list[i].to(args.device)
                    input_ids = input_ids_list[i].to(args.device)
                    attention_mask = attention_mask_list[i].to(args.device)
                    zeta_j, p_j, q_j, T = model(imgs, input_ids, attention_mask)
                    scores.append(zeta_j)
                scores = torch.stack(scores, dim=1)
                loss = F.cross_entropy(scores, answers.to(args.device))
                train_loss += loss.item() * batch_size
            else:
                imgs, input_ids, attention_mask, lengths, labels = batch
                imgs = imgs.to(args.device)
                input_ids = input_ids.to(args.device)
                attention_mask = attention_mask.to(args.device)
                labels = labels.to(args.device)
                zeta_j, p_j, q_j, T = model(imgs, input_ids, attention_mask)
                loss, L_com, L_corr = model.compute_loss(zeta_j, p_j, q_j, T, labels)
                train_loss += loss.item() * imgs.size(0)
                train_com += L_com.item() * imgs.size(0)
                train_corr += L_corr.item() * imgs.size(0)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        train_loss /= len(train_loader.dataset)
        train_com /= len(train_loader.dataset)
        train_corr /= len(train_loader.dataset)

        model.eval()
        val_loss, val_com, val_corr = 0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                if args.task == 'fitb':
                    imgs_list, input_ids_list, attention_mask_list, lengths, answers = batch
                    scores = []
                    for i in range(4):
                        imgs = imgs_list[i].to(args.device)
                        input_ids = input_ids_list[i].to(args.device)
                        attention_mask = attention_mask_list[i].to(args.device)
                        zeta_j, _, _, _ = model(imgs, input_ids, attention_mask)
                        scores.append(zeta_j)
                    scores = torch.stack(scores, dim=1)
                    loss = F.cross_entropy(scores, answers.to(args.device))
                    val_loss += loss.item() * len(lengths)
                else:
                    imgs, input_ids, attention_mask, lengths, labels = batch
                    imgs = imgs.to(args.device)
                    input_ids = input_ids.to(args.device)
                    attention_mask = attention_mask.to(args.device)
                    labels = labels.to(args.device)
                    zeta_j, p_j, q_j, T = model(imgs, input_ids, attention_mask)
                    loss, L_com, L_corr = model.compute_loss(zeta_j, p_j, q_j, T, labels)
                    val_loss += loss.item() * imgs.size(0)
                    val_com += L_com.item() * imgs.size(0)
                    val_corr += L_corr.item() * imgs.size(0)

        val_loss /= len(val_loader.dataset)
        val_com /= len(val_loader.dataset)
        val_corr /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{args.num_epochs}, Train Loss: {train_loss:.4f}, "
              f"Train L_com: {train_com:.4f}, Train L_corr: {train_corr:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val L_com: {val_com:.4f}, Val L_corr: {val_corr:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')

    return best_val_loss

def evaluate_fitb(model, test_loader, args):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs_list, input_ids_list, attention_mask_list, lengths, answers in test_loader:
            best_idx, scores = model.recommend_fitb(imgs_list, input_ids_list, attention_mask_list)
            correct += (best_idx.cpu() == answers).sum().item()
            total += answers.size(0)
    accuracy = correct / total
    return accuracy

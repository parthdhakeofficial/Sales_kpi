# TODO

## 1) Confirm root error
- [ ] Ask user to paste Flask traceback from terminal.

## 2) Confirm data schema mismatch
- [ ] Update code to normalize uploaded CSV columns to what ML expects.

## 3) Implement column mapping (planned)
- [ ] In `upload()` rename known alternatives:
  - customer_type -> customer_name
  - sales_amount -> amount
  - sale_date -> date
- [ ] Ensure mapping happens before `save_df(df)`.
- [ ] Make ML robust if columns are missing (guard + better flash message).

## 4) Testing
- [ ] Re-upload the dataset that currently fails.
- [ ] Hit `/ml` and verify clustering runs.
- [ ] Hit `/ml` again after restart to verify LR training behaves.


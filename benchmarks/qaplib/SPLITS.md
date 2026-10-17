# QAPLIB splits

Seed 20261017. Stratified by family and size (blocks of 5 sorted by family, n: 2 train / 1 dev / 2 held-out).
Held-out is frozen: `splits.py` refuses to overwrite a `splits.json` whose held-out set differs.

Held-out hash (sha256 over `name:file_sha256`): `079357c5a1598945a12455ff9c9227b21894d8bba1cddbe54bb823dac1f4d84b`

## train (18 instances, n 15..81)

| name | family | n | BKS | proven optimal | sha256 (.dat) |
|---|---|---|---|---|---|
| chr18b | chr | 18 | 1534 | yes | `7400d871613c545e` |
| chr20b | chr | 20 | 2298 | yes | `5982959046627fd5` |
| els19 | els | 19 | 17212548 | yes | `649de7f196439afd` |
| kra30b | kra | 30 | 91420 | yes | `e497b762f1253e08` |
| lipa20a | lipa | 20 | 3683 | yes | `3e0346b250c7060b` |
| lipa70a | lipa | 70 | 169755 | yes | `9291885647c09b68` |
| nug17 | nug | 17 | 1732 | yes | `d5cbe356fe813882` |
| nug25 | nug | 25 | 3744 | yes | `c8c62f0ec01dd5c0` |
| rou15 | rou | 15 | 354210 | yes | `7631c0017c53061e` |
| sko56 | sko | 56 | 34458 | no | `581b41269fd70985` |
| sko81 | sko | 81 | 90998 | no | `40aae61081c6076a` |
| ste36a | ste | 36 | 9526 | yes | `0dc451cb75985a53` |
| tai20b | tai-b | 20 | 122455319 | yes | `99efd6bc04aae2f8` |
| tai35a | tai-a | 35 | 2422002 | no | `78ac8afe13129e90` |
| tai50a | tai-a | 50 | 4938796 | no | `3ff7b0cc5cbc28d0` |
| tai80a | tai-a | 80 | 13499184 | no | `99a401c93d729442` |
| tai80b | tai-b | 80 | 818415043 | no | `cea03ad8f4fd1755` |
| tho40 | tho | 40 | 240516 | no | `2701531cf5f9e03d` |

## dev (9 instances, n 20..100)

| name | family | n | BKS | proven optimal | sha256 (.dat) |
|---|---|---|---|---|---|
| chr22a | chr | 22 | 6156 | yes | `cc60a1117a2bf677` |
| had20 | had | 20 | 6922 | yes | `7fcf5045bd3918f5` |
| lipa40b | lipa | 40 | 476581 | yes | `da4b86feee53bc81` |
| nug20 | nug | 20 | 2570 | yes | `a82316ada09e9c8d` |
| sko100a | sko | 100 | 152002 | no | `bd25a54fe4761df0` |
| sko42 | sko | 42 | 15812 | no | `fd38875376426302` |
| tai50b | tai-b | 50 | 458821517 | no | `4977aa11a530a069` |
| tai60a | tai-a | 60 | 7205962 | no | `b32cae60a3539c78` |
| tho30 | tho | 30 | 149936 | yes | `270072ddc0b14bcd` |

## heldout (18 instances, n 15..100)

| name | family | n | BKS | proven optimal | sha256 (.dat) |
|---|---|---|---|---|---|
| bur26a | bur | 26 | 5426670 | yes | `bcc4b7ddd80b49b4` |
| chr15c | chr | 15 | 9504 | yes | `61b307b61269d6a7` |
| chr25a | chr | 25 | 3796 | yes | `e840425ca03bd148` |
| kra30a | kra | 30 | 88900 | yes | `edc54d8bfc77d054` |
| lipa50a | lipa | 50 | 62093 | yes | `c4691880bd5a8910` |
| lipa90a | lipa | 90 | 360630 | yes | `8448c98c34e4811f` |
| nug22 | nug | 22 | 3596 | yes | `406aa5df910fbbea` |
| nug30 | nug | 30 | 6124 | yes | `7c8e1001b1fb13e9` |
| rou20 | rou | 20 | 725522 | yes | `7a2c7acf9ee38b25` |
| scr20 | scr | 20 | 110030 | yes | `2511fcaae6de80b2` |
| sko64 | sko | 64 | 48498 | no | `fc1a95a121c50da5` |
| ste36c | ste | 36 | 8239110 | yes | `9071556e17b7461d` |
| tai100a | tai-a | 100 | 21052466 | no | `de053a8df4044c71` |
| tai15a | tai-a | 15 | 388214 | yes | `374f8225097d01ab` |
| tai20a | tai-a | 20 | 703482 | yes | `39eeebc193ffa134` |
| tai35b | tai-b | 35 | 283315445 | no | `7082b75870365c9e` |
| wil100 | wil | 100 | 273038 | no | `9477abe6459def2f` |
| wil50 | wil | 50 | 48816 | no | `ad67fae8ff5e78aa` |

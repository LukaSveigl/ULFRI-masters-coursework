import { blob, integer, sqliteTable, text, uniqueIndex } from "drizzle-orm/sqlite-core";

export const user = sqliteTable("user", {
    id: blob("id", { mode: "buffer" }).primaryKey(),
    displayName: text("display_name").notNull(),
    avatar: blob("avatar", { mode: "buffer" }),
    avatarModified: integer("avatar_modified", { mode: "timestamp" }).notNull(),
    username: text("username").notNull().unique(),
    passwordHash: text("password_hash").notNull()
});

export const session = sqliteTable("session", {
    id: blob("id", { mode: "buffer" }).primaryKey(),
    userId: blob("user_id", { mode: "buffer" })
        .notNull()
        .references(() => user.id, { onDelete: "cascade", onUpdate: "cascade" }),
    expiresAt: integer("expires_at", { mode: "timestamp" }).notNull()
});

export const friend = sqliteTable(
    "friend",
    {
        id: blob("id", { mode: "buffer" }).primaryKey(),
        user1Id: blob("user1_id", { mode: "buffer" })
            .notNull()
            .references(() => user.id, { onDelete: "cascade", onUpdate: "cascade" }),
        user2Id: blob("user2_id", { mode: "buffer" })
            .notNull()
            .references(() => user.id, { onDelete: "cascade", onUpdate: "cascade" })
    },
    (table) => ({
        uqFriend: uniqueIndex("uq_friend").on(table.user1Id, table.user2Id)
    })
);

export const group = sqliteTable("group", {
    id: blob("id", { mode: "buffer" }).primaryKey(),
    displayName: text("display_name").notNull(),
    avatar: blob("avatar", { mode: "buffer" }),
    avatarModified: integer("avatar_modified", { mode: "timestamp" }).notNull(),
    adminUserId: blob("admin_user_id", { mode: "buffer" }).references(() => user.id, {
        onDelete: "set null",
        onUpdate: "cascade"
    })
});

export const groupMember = sqliteTable(
    "group_member",
    {
        id: blob("id", { mode: "buffer" }).primaryKey(),
        userId: blob("user_id", { mode: "buffer" })
            .references(() => user.id, { onDelete: "cascade", onUpdate: "cascade" })
            .notNull(),
        groupId: blob("group_id", { mode: "buffer" })
            .references(() => group.id, { onDelete: "cascade", onUpdate: "cascade" })
            .notNull()
    },
    (table) => ({
        uqMembership: uniqueIndex("uq_membership").on(table.userId, table.groupId)
    })
);

export const loan = sqliteTable("loan", {
    id: blob("id", { mode: "buffer" }).primaryKey(),
    issuerUserId: blob("issuer_user_id", { mode: "buffer" }).references(() => user.id, {
        onDelete: "set null",
        onUpdate: "cascade"
    }),
    groupId: blob("group_id", { mode: "buffer" }).references(() => group.id, {
        onDelete: "set null",
        onUpdate: "cascade"
    }),
    createdOn: integer("created_on", { mode: "timestamp" }).notNull()
});

export const loanPayment = sqliteTable(
    "loan_payment",
    {
        id: blob("id", { mode: "buffer" }).primaryKey(),
        loanId: blob("loan_id", { mode: "buffer" })
            .references(() => loan.id, { onDelete: "cascade", onUpdate: "cascade" })
            .notNull(),
        userId: blob("user_id", { mode: "buffer" }).references(() => user.id, {
            onDelete: "set null",
            onUpdate: "cascade"
        }),
        value: integer("value").notNull(),
        valuePaid: integer("value_paid").notNull().default(0),
        valuePaidConfirmed: integer("value_paid_confirmed").notNull().default(0)
    },
    (table) => ({ uqLoan: uniqueIndex("uq_loan").on(table.loanId, table.userId) })
);
